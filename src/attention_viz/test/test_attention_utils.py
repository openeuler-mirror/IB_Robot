#!/usr/bin/env python3

from pathlib import Path

import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None


def require_torch():
    if torch is None:
        pytest.skip("torch is not available in this test environment")


def test_attention_data_round_trip_preserves_shape_and_topics():
    require_torch()
    from attention_viz.utils import attention_data_to_msg, msg_to_attention_data

    weights = torch.arange(2 * 1 * 3 * 4 * 7, dtype=torch.float32).reshape(
        2, 1, 3, 4, 7
    )
    payload = {
        "attn_weights": weights,
        "feature_map_size": (2, 3),
        "camera_keys": ["observation.images.top", "observation.images.wrist"],
        "camera_topics": ["/camera/top/image_raw", "/camera/wrist/image_raw"],
        "num_non_image_tokens": 1,
        "average_heads": True,
        "blend_alpha": 0.25,
    }

    msg = attention_data_to_msg(payload, stamp_sec=1, stamp_nanosec=2)
    restored = msg_to_attention_data(msg)

    assert restored["attention_weights"].shape == weights.shape
    assert torch.equal(restored["attention_weights"], weights)
    assert restored["camera_topics"] == payload["camera_topics"]
    assert restored["feature_map_size"] == payload["feature_map_size"]


def test_extract_attention_per_camera_supports_five_dimensional_input():
    require_torch()
    from attention_viz.utils import extract_attention_per_camera

    weights = torch.arange(1 * 1 * 2 * 2 * 5, dtype=torch.float32).reshape(
        1, 1, 2, 2, 5
    )
    camera_keys = ["observation.images.top", "observation.images.wrist"]

    per_camera = extract_attention_per_camera(
        weights,
        camera_keys=camera_keys,
        feature_map_size=(1, 2),
        num_non_image_tokens=1,
        query_idx=1,
        batch_idx=0,
        layer_idx=0,
        average_heads=False,
    )

    expected = weights[0, 0, 0, 1, 1:]
    assert torch.equal(per_camera[camera_keys[0]], expected[:2])
    assert torch.equal(per_camera[camera_keys[1]], expected[2:])


def test_camera_topic_map_prefers_explicit_overrides():
    from attention_viz.utils import (
        build_camera_topic_map,
        default_camera_topic_from_key,
    )

    camera_keys = ["observation.images.top", "observation.images.wrist"]

    topic_map = build_camera_topic_map(
        camera_keys,
        message_camera_topics=["/camera/top/image_raw", ""],
        configured_camera_topics=[
            "observation.images.wrist:=/custom/wrist/image_raw"
        ],
    )

    assert topic_map["observation.images.top"] == "/camera/top/image_raw"
    assert topic_map["observation.images.wrist"] == "/custom/wrist/image_raw"
    assert (
        default_camera_topic_from_key("observation.images.front")
        == "/camera/front/image_raw"
    )


def test_visualization_mode_supports_interactive_alias():
    from attention_viz.utils import normalize_visualization_mode

    assert normalize_visualization_mode("interactive") == "realtime"
    assert normalize_visualization_mode("REALTIME") == "realtime"
    assert normalize_visualization_mode("file") == "file"
    assert normalize_visualization_mode("unknown") == "file"


def test_stacked_attention_hook_reset_clears_existing_storage_in_place():
    require_torch()
    from attention_viz.attention_hook import StackedAttentionHook

    hook = StackedAttentionHook()
    hook._all_layer_weights.append(torch.tensor([1.0]))
    existing_ref = hook._all_layer_weights
    hook._latest_weights = torch.tensor([1.0])

    hook.reset_for_new_inference()

    assert hook._all_layer_weights is existing_ref
    assert hook._all_layer_weights == []
    assert hook.get_latest() is None


def test_stacked_attention_hook_publishes_only_after_all_layers():
    require_torch()
    from attention_viz.attention_hook import StackedAttentionHook

    hook = StackedAttentionHook()
    hook._expected_layer_count = 2
    first_layer_hook = hook._make_hook_fn()
    second_layer_hook = hook._make_hook_fn()

    first_layer_hook(None, None, (None, torch.ones(1, 2, 3)))
    assert hook.get_latest() is None

    second_layer_hook(None, None, (None, torch.full((1, 2, 3), 2.0)))
    latest = hook.get_latest()

    assert latest is not None
    assert latest["attn_weights"].shape == (2, 1, 2, 3)
    assert torch.equal(latest["attn_weights"][0], torch.ones(1, 2, 3))
    assert torch.equal(latest["attn_weights"][1], torch.full((1, 2, 3), 2.0))


def test_transformer_attention_masks_follow_act_token_layout():
    require_torch()
    from attention_viz.attention_masking import build_transformer_attention_masks

    feature_masks = {
        "observation.images.top": torch.tensor([True, False]),
    }

    decoder_mask, encoder_mask = build_transformer_attention_masks(
        feature_masks,
        camera_keys=["observation.images.top", "observation.images.wrist"],
        feature_map_size=(1, 2),
        num_non_image_tokens=2,
        chunk_size=3,
        batch_size=1,
        device="cpu",
    )

    expected = torch.tensor([False, False, True, False, False, False])
    assert torch.equal(decoder_mask, expected.unsqueeze(0).expand(3, -1))
    assert torch.equal(encoder_mask, expected.unsqueeze(0))


def test_tensor_image_to_rgb_handles_chw_float_tensors():
    require_torch()
    from attention_viz.attention_masking import tensor_image_to_rgb

    image = torch.zeros(1, 3, 2, 2)
    image[:, 0] = 0.5
    image[:, 1] = 1.0

    rgb = tensor_image_to_rgb(image)

    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype.name == "uint8"
    assert rgb[0, 0, 0] == 127
    assert rgb[0, 0, 1] == 255


def test_tensor_image_to_rgb_handles_hwc_numpy_float_images():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from attention_viz.attention_masking import tensor_image_to_rgb

    image = np.zeros((2, 2, 3), dtype=np.float32)
    image[..., 0] = 0.5
    image[..., 1] = 1.0

    rgb = tensor_image_to_rgb(image)

    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype.name == "uint8"
    assert rgb[0, 0, 0] == 127
    assert rgb[0, 0, 1] == 255


def test_attention_hook_injects_decoder_attention_mask():
    require_torch()
    from attention_viz.attention_hook import StackedAttentionHook

    class DummyConfig:
        image_features = {"observation.images.top": object()}

    class DummyModel:
        def __init__(self):
            self.decoder = torch.nn.Module()
            self.decoder.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.decoder.layers[0].multihead_attn = torch.nn.MultiheadAttention(
                4,
                1,
            )
            self.encoder = torch.nn.Module()
            self.encoder.layers = torch.nn.ModuleList([])

    class DummyPolicy:
        def __init__(self):
            self.config = DummyConfig()
            self.model = DummyModel()

    policy = DummyPolicy()
    hook = StackedAttentionHook()
    assert hook.install(policy)
    hook.set_attention_masks(torch.tensor([[False, True]]), None)

    query = torch.zeros(1, 1, 4)
    key = torch.zeros(2, 1, 4)
    _output, weights = policy.model.decoder.layers[0].multihead_attn(
        query,
        key,
        key,
    )

    hook.uninstall()
    assert weights.shape == (1, 1, 1, 2)
    assert weights[0, 0, 0, 1] == 0


def test_attention_hook_injects_encoder_key_padding_mask():
    require_torch()
    from attention_viz.attention_hook import StackedAttentionHook

    class DummyConfig:
        image_features = {"observation.images.top": object()}

    class DummyModel:
        def __init__(self):
            self.decoder = torch.nn.Module()
            self.decoder.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.decoder.layers[0].multihead_attn = torch.nn.MultiheadAttention(
                4,
                1,
            )
            self.encoder = torch.nn.Module()
            self.encoder.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.encoder.layers[0].self_attn = torch.nn.MultiheadAttention(4, 1)

    class DummyPolicy:
        def __init__(self):
            self.config = DummyConfig()
            self.model = DummyModel()

    policy = DummyPolicy()
    hook = StackedAttentionHook()
    assert hook.install(policy)
    hook.set_attention_masks(None, torch.tensor([[False, True]]))

    query = torch.zeros(2, 1, 4)
    _output, weights = policy.model.encoder.layers[0].self_attn(
        query,
        query,
        query,
        need_weights=True,
        average_attn_weights=False,
    )

    hook.uninstall()
    assert weights.shape == (1, 1, 2, 2)
    assert torch.all(weights[..., 1] == 0)


def test_feature_mask_conversion_resizes_pixel_masks():
    require_torch()
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from attention_viz.attention_masking import create_feature_masks_from_pixel_masks

    pixel_masks = {
        "observation.images.top": np.array(
            [
                [255, 255, 0, 0],
                [255, 255, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
    }

    feature_masks = create_feature_masks_from_pixel_masks(
        pixel_masks,
        feature_map_size=(2, 2),
    )

    assert torch.equal(
        feature_masks["observation.images.top"],
        torch.tensor([True, False, False, False]),
    )


def test_realtime_preprocess_uses_message_non_image_token_count():
    require_torch()
    pytest.importorskip("cv2")
    pytest.importorskip("matplotlib")
    from attention_viz.visualization_core import RealTimeVisualizer

    visualizer = RealTimeVisualizer(
        camera_keys=["observation.images.top", "observation.images.wrist"],
        queries_to_visualize=[0],
        layer_idx=0,
        batch_idx=0,
        average_heads=False,
        blend_alpha=0.4,
    )
    weights = torch.arange(1 * 1 * 1 * 1 * 6, dtype=torch.float32).reshape(
        1, 1, 1, 1, 6
    )

    per_query = visualizer._preprocess_attn(
        weights,
        feature_map_size=(1, 1),
        num_non_image_tokens=2,
    )

    assert torch.equal(
        per_query[(0, "observation.images.top")],
        torch.tensor([2.0]),
    )
    assert torch.equal(
        per_query[(0, "observation.images.wrist")],
        torch.tensor([3.0]),
    )


def test_visualization_core_keeps_non_agg_backend_when_display_available(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    source = Path(__file__).resolve().parents[1].joinpath(
        "attention_viz", "visualization_core.py"
    ).read_text(encoding="utf-8")
    pyplot_import = source.index("import matplotlib.pyplot as plt")
    backend_switch = source.index('matplotlib.use("Agg")')
    assert backend_switch < pyplot_import


def test_visualization_core_uses_public_rectangle_class():
    source = Path(__file__).resolve().parents[1].joinpath(
        "attention_viz", "visualization_core.py"
    ).read_text(encoding="utf-8")

    assert "from matplotlib.patches import Rectangle" in source
    assert "plt.Rectangle" not in source


def test_visualize_attention_maps_handles_hook_axis_order(tmp_path):
    require_torch()
    pytest.importorskip("cv2")
    pytest.importorskip("matplotlib")
    np = pytest.importorskip("numpy")

    from attention_viz.visualization_core import visualize_attention_maps

    camera_keys = ["observation.images.top", "observation.images.wrist"]
    weights = torch.zeros(1, 1, 1, 81, 9, dtype=torch.float32)
    for query_idx in [0, 20, 80]:
        weights[0, 0, 0, query_idx, 1:] = torch.linspace(0.1, 1.0, 8)
    original_images = {
        key: np.full((32, 32, 3), 127, dtype=np.uint8)
        for key in camera_keys
    }

    visualize_attention_maps(
        original_images,
        weights,
        camera_keys,
        feature_map_size=(2, 2),
        step_counter=1,
        save_dir=str(tmp_path),
        queries_to_visualize=[0, 20, 80],
        num_non_image_tokens=1,
    )

    assert len(list(tmp_path.rglob("*_attn.jpg"))) == 6
    assert (tmp_path / "observation_images_top" / "action_020").is_dir()


def test_visualize_attention_maps_skips_missing_camera_image(tmp_path):
    require_torch()
    pytest.importorskip("cv2")
    pytest.importorskip("matplotlib")
    np = pytest.importorskip("numpy")

    from attention_viz.visualization_core import visualize_attention_maps

    camera_keys = ["observation.images.top", "observation.images.wrist"]
    weights = torch.ones(1, 1, 1, 1, 5, dtype=torch.float32)
    original_images = {
        "observation.images.top": np.full((32, 32, 3), 127, dtype=np.uint8)
    }

    visualize_attention_maps(
        original_images,
        weights,
        camera_keys,
        feature_map_size=(1, 2),
        step_counter=1,
        save_dir=str(tmp_path),
        queries_to_visualize=[0],
        num_non_image_tokens=1,
    )

    assert len(list(tmp_path.rglob("*_attn.jpg"))) == 1
    assert (tmp_path / "observation_images_top" / "action_000").is_dir()

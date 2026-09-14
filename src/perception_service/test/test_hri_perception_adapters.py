"""Pin the preprocessing geometry of the HRI (YOLOX -> PEAR) adapters.

Every constant asserted here is part of an external model bundle's declared
contract, not an implementation detail: the exported graphs were traced with
these exact conventions, and a silent change produces plausible-looking numbers
that are wrong. The offline handoff comparison catches that, these tests catch
it earlier.
"""

import numpy as np
import pytest

from inference_service.unified_runtime import (
    ModelResult,
    OutcomeEvidence,
    RuntimeLatency,
)
from perception_service.pear_adapter import PearParameterAdapter
from perception_service.yolox_adapter import YoloXAdapter


def _result(outputs):
    return ModelResult(
        outputs=outputs,
        latency=RuntimeLatency(total_ms=1.0, backend_ms=1.0),
        evidence=OutcomeEvidence.completed("adaptation"),
    )


def _coordinate_ramp(height: int, width: int) -> np.ndarray:
    """RGB image whose R channel encodes x and G channel encodes y."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(width, dtype=np.uint8)[None, :]
    image[:, :, 1] = np.arange(height, dtype=np.uint8)[:, None]
    return image


# --------------------------------------------------------------------------
# PEAR crop geometry
# --------------------------------------------------------------------------


def test_pear_crop_constants_match_the_model_card() -> None:
    assert PearParameterAdapter._CROP_SIZE == 256
    assert PearParameterAdapter._CROP_MARGIN == 1.25


def _edge_at(height: int, width: int, x_edge: int, y_edge: int) -> np.ndarray:
    """RGB image with a hard step edge in R (at x_edge) and in G (at y_edge).

    A ramp is useless for detecting a sub-pixel sampling error: the interpolated
    value rounds back to the same uint8. A step edge does not.
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, x_edge:, 0] = 255
    image[y_edge:, :, 1] = 255
    return image


def test_pear_crop_maps_square_corners_onto_the_last_pixel_not_past_it() -> None:
    """The affine target corner is 255, not 256.

    cv2.getAffineTransform maps pixel *centres*. Using _CROP_SIZE as the target
    corner rescales every crop by 256/255, which shifts the regressed pose by an
    amount no shape or finiteness check would ever notice.

    The box below is chosen so the 1.25-margin square lands on integer source
    coordinates (100 to 200). With the correct 255 target the last crop column
    samples source x=200 and lands on the bright side of the step; with a 256
    target it samples x=199.6 and interpolates across the edge instead.
    """
    image = _edge_at(300, 300, x_edge=200, y_edge=200)
    # w = h = 80 -> half side = 80 * 1.25 / 2 = 50, square = [100, 200]
    box = np.array([110.0, 110.0, 190.0, 190.0], dtype=np.float32)

    tensor = PearParameterAdapter().preprocess((image, [box]))["pear.input"]
    assert tensor.shape == (1, 3, 256, 256)
    assert tensor.dtype == np.float32

    # NCHW is BGR, so channel 2 is R (steps at x) and channel 1 is G (steps at y).
    x_plane, y_plane = tensor[0, 2], tensor[0, 1]
    assert x_plane[0, 255] == pytest.approx(1.0, abs=1e-6)
    assert y_plane[255, 0] == pytest.approx(1.0, abs=1e-6)
    # The column/row just inside the edge is still on the dark side.
    assert x_plane[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert y_plane[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_pear_crop_uses_the_longer_box_side_so_the_crop_stays_square() -> None:
    image = _coordinate_ramp(300, 300)
    # A wide box: side must come from w (80), not h (20), and stay centred.
    box = np.array([110.0, 140.0, 190.0, 160.0], dtype=np.float32)

    x_plane = PearParameterAdapter().preprocess((image, [box]))["pear.input"][0, 2]
    y_plane = PearParameterAdapter().preprocess((image, [box]))["pear.input"][0, 1]
    assert x_plane[0, 0] == pytest.approx(100.0 / 255.0, abs=1e-6)
    assert x_plane[0, 255] == pytest.approx(200.0 / 255.0, abs=1e-6)
    # Vertical extent is the same 100 px square, centred on cy = 150.
    assert y_plane[0, 0] == pytest.approx(100.0 / 255.0, abs=1e-6)
    assert y_plane[255, 0] == pytest.approx(200.0 / 255.0, abs=1e-6)


def test_pear_crop_pads_outside_the_frame_with_zeros() -> None:
    image = np.full((300, 300, 3), 200, dtype=np.uint8)
    # Square runs from -50 to 50: the left half of the crop falls outside.
    box = np.array([-40.0, -40.0, 40.0, 40.0], dtype=np.float32)

    tensor = PearParameterAdapter().preprocess((image, [box]))["pear.input"]
    assert tensor[0, :, 0, 0] == pytest.approx(0.0, abs=1e-6)
    assert tensor[0, :, 255, 255] == pytest.approx(200.0 / 255.0, abs=1e-6)


def test_pear_crop_is_taken_in_source_coordinates_not_letterbox_space() -> None:
    """The same person in a larger frame must yield the same crop."""
    small = _coordinate_ramp(300, 300)
    large = np.zeros((900, 1600, 3), dtype=np.uint8)
    large[:300, :300] = small
    box = np.array([110.0, 110.0, 190.0, 190.0], dtype=np.float32)

    adapter = PearParameterAdapter()
    from_small = adapter.preprocess((small, [box]))["pear.input"]
    from_large = adapter.preprocess((large, [box]))["pear.input"]
    np.testing.assert_array_equal(from_small, from_large)


@pytest.mark.parametrize("count", [0, 2])
def test_pear_rejects_anything_other_than_one_box(count: int) -> None:
    image = _coordinate_ramp(300, 300)
    boxes = [np.array([110.0, 110.0, 190.0, 190.0], dtype=np.float32)] * count
    with pytest.raises(ValueError, match="exactly one person crop"):
        PearParameterAdapter().preprocess((image, boxes))


@pytest.mark.parametrize(
    "box, match",
    [
        ([150.0, 150.0, 150.0, 150.0], "positive extent"),
        ([200.0, 200.0, 100.0, 100.0], "positive extent"),
        ([float("nan")] * 4, "finite"),
        ([100.0, 100.0, float("inf"), 200.0], "finite"),
    ],
)
def test_pear_rejects_degenerate_boxes_instead_of_cropping_flat_padding(box, match) -> None:
    """cv2 does not raise on these - it returns a uniform patch that looks healthy.

    PEAR then regresses a pose from flat padding, and the shape, finiteness and
    success checks downstream all pass. The only place this can be caught is here.
    """
    image = _coordinate_ramp(300, 300)
    with pytest.raises(ValueError, match=match):
        PearParameterAdapter().preprocess((image, [np.asarray(box, dtype=np.float32)]))


def test_pear_rejects_non_rgb_uint8_images() -> None:
    box = np.array([110.0, 110.0, 190.0, 190.0], dtype=np.float32)
    with pytest.raises(ValueError, match="RGB uint8"):
        PearParameterAdapter().preprocess((np.zeros((300, 300, 3), np.float32), [box]))


# --------------------------------------------------------------------------
# YOLOX letterbox and decode
# --------------------------------------------------------------------------


def test_yolox_feeds_the_graph_bgr_because_it_was_traced_on_cv2_imread() -> None:
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    image[:, :, 0] = 10  # R
    image[:, :, 1] = 20  # G
    image[:, :, 2] = 30  # B

    tensor = YoloXAdapter().preprocess(image)["observation.image"]
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor[0, 0, 0, 0] == pytest.approx(30.0)  # B first
    assert tensor[0, 1, 0, 0] == pytest.approx(20.0)
    assert tensor[0, 2, 0, 0] == pytest.approx(10.0)


def test_yolox_letterbox_pastes_top_left_and_pads_with_114() -> None:
    image = np.full((360, 640, 3), 7, dtype=np.uint8)

    tensor = YoloXAdapter().preprocess(image)["observation.image"]
    # ratio = min(640/360, 640/640) = 1.0, so content occupies rows 0..359.
    assert tensor[0, 0, 359, 0] == pytest.approx(7.0)
    assert tensor[0, 0, 360, 0] == pytest.approx(114.0)
    assert tensor[0, 0, 639, 639] == pytest.approx(114.0)


def test_yolox_decode_produces_xyxy_as_centre_plus_minus_half_size() -> None:
    """Upstream YOLOX writes x2 = cx + w/2, not x1 + w.

    The two forms are not the same in float32. The residual is tiny - about
    1e-6 px here, far below anything visible - but it is enough to break a
    bit-identical comparison against the reference implementation, which is the
    gate the bundle was validated under. These raw values were picked because
    they are one of the inputs where the two forms diverge; on round numbers
    like 0.5 they agree exactly and the test would prove nothing.
    """
    raw_centre = np.float32(0.84016883)
    raw_log_size = np.float32(-0.4379713)

    raw = np.zeros((1, 8400, 85), dtype=np.float32)
    # Anchor 0 is stride 8, grid (0, 0), so the grid offset drops out.
    raw[0, 0, :4] = (raw_centre, raw_centre, raw_log_size, raw_log_size)
    raw[0, 0, 4] = 0.9  # objectness
    raw[0, 0, 5] = 0.8  # person class score

    detections = YoloXAdapter().postprocess(_result({"yolox.raw": raw}), image_shape=(640, 640))
    assert len(detections) == 1
    detection = detections[0]
    assert detection.label == "person"
    assert detection.confidence == pytest.approx(0.72, abs=1e-6)

    stride = np.float32(8.0)
    centre = np.float32(raw_centre * stride)
    half = np.float32(np.float32(np.exp(raw_log_size)) * stride / np.float32(2.0))
    upstream = np.float32(centre + half)
    rearranged = np.float32(np.float32(centre - half) + np.float32(half * np.float32(2.0)))
    assert upstream != rearranged, "these inputs no longer discriminate the two forms"

    assert detection.bbox_xyxy[2] == upstream
    assert detection.bbox_xyxy[3] == upstream
    assert detection.bbox_xyxy[0] == np.float32(centre - half)


def test_yolox_maps_boxes_back_to_source_coordinates_by_dividing_by_ratio() -> None:
    raw = np.zeros((1, 8400, 85), dtype=np.float32)
    raw[0, 0, :4] = (0.5, 0.5, np.log(10.0), np.log(10.0))
    raw[0, 0, 4] = 1.0
    raw[0, 0, 5] = 1.0

    # ratio = min(640/1280, 640/1280) = 0.5, so source boxes are twice as large.
    at_half = YoloXAdapter().postprocess(_result({"yolox.raw": raw}), image_shape=(1280, 1280))[0].bbox_xyxy
    at_one = YoloXAdapter().postprocess(_result({"yolox.raw": raw}), image_shape=(640, 640))[0].bbox_xyxy
    np.testing.assert_allclose(at_half, at_one * 2.0, rtol=1e-6)

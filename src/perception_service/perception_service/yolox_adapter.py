"""YOLOX-X person detection adapter for the compiled Ascend bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from inference_service.unified_runtime import ModelResult

from .adapter_support import output_tensor, read_adapter_identity
from .perception_adapter import AdapterIdentity, PerceptionAdapter


@dataclass(frozen=True)
class YoloXDetection:
    label: str
    confidence: float
    bbox_xyxy: np.ndarray
    mask: np.ndarray | None = None


class YoloXAdapter(PerceptionAdapter):
    """YOLOX-X person detector adapter for the compiled [1,8400,85] graph."""

    identity = AdapterIdentity(
        "yolox_person",
        "yolox-rgb-letterbox640-v1",
        "person-bboxes-confidence-v1",
        frozenset({"ascend_310p"}),
        operation="detect",
    )
    compiled_abi_finalized = True

    @classmethod
    def from_bundle(cls, bundle_root: str | Path, _identity=None, *, model=None, deployment=None):
        del deployment, model
        read_adapter_identity(Path(bundle_root), cls.identity)
        return cls()

    def preprocess(self, image_rgb: np.ndarray) -> dict[str, np.ndarray]:
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("image must be an RGB uint8 HxWx3 array")
        # The exported graph was traced on the upstream YOLOX demo path, which feeds
        # cv2.imread order (BGR). Sending RGB is silently wrong rather than fatal: box
        # centres shift by up to 81 px, and the body joints derived from them by up to
        # 24 cm.
        image = np.ascontiguousarray(image[:, :, ::-1])
        height, width = image.shape[:2]
        ratio = min(640.0 / height, 640.0 / width)
        resized = cv2.resize(image, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32)[None]
        return {"observation.image": tensor}

    @staticmethod
    def _decode(
        raw: np.ndarray, image_shape: tuple[int, int], ratio: float, nms_threshold: float = 0.65
    ) -> list[YoloXDetection]:
        values = np.asarray(raw, dtype=np.float32)
        if values.shape == (1, 8400, 85):
            values = values[0]
        if values.shape != (8400, 85):
            raise RuntimeError(f"YOLOX output must have shape [1,8400,85], got {values.shape}")
        grids = []
        strides = []
        for stride in (8, 16, 32):
            size = 640 // stride
            yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
            grids.append(np.stack((xx, yy), axis=-1).reshape(-1, 2))
            strides.append(np.full(size * size, stride, dtype=np.float32))
        grid = np.concatenate(grids, axis=0).astype(np.float32)
        stride = np.concatenate(strides)
        raw_boxes = values[:, :4]
        centers = (raw_boxes[:, :2] + grid) * stride[:, None]
        half_sizes = np.exp(np.clip(raw_boxes[:, 2:], -20.0, 20.0)) * stride[:, None] / 2.0
        # Keep upstream YOLOX's cx +/- w/2 form. Deriving x2 as x1 + w is not the same
        # computation in float32: the two forms disagree on roughly one anchor in ten,
        # by ~1e-6 px. Invisible in the output, but enough to break the bit-identical
        # comparison the bundle was validated under.
        boxes = np.concatenate((centers - half_sizes, centers + half_sizes), axis=1)
        scores = values[:, 4] * values[:, 5]
        keep = scores >= 0.01
        boxes, scores = boxes[keep], scores[keep]
        if not len(boxes):
            return []
        order = np.argsort(-scores)
        selected_list = []
        while len(order):
            index = int(order[0])
            selected_list.append(index)
            if len(order) == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(boxes[index, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[index, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[index, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[index, 3], boxes[rest, 3])
            intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
            union = area[index] + area[rest] - intersection
            order = rest[(intersection / np.maximum(union, 1e-6)) <= float(nms_threshold)]
        selected = np.asarray(selected_list, dtype=np.int64)
        height, width = image_shape
        result = []
        for index in selected:
            box = boxes[index] / float(ratio)
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            result.append(YoloXDetection("person", float(scores[index]), box.astype(np.float32)))
        return sorted(result, key=lambda item: -item.confidence)

    def postprocess(
        self, result: ModelResult, *, image_shape=None, nms_threshold=0.65, **_options
    ) -> list[YoloXDetection]:
        if image_shape is None or len(image_shape) != 2:
            raise ValueError("YOLOX postprocess requires source image shape")
        raw = output_tensor(result, "yolox.raw", np.float32)
        height, width = image_shape
        ratio = min(640.0 / height, 640.0 / width)
        return self._decode(raw, (height, width), ratio, float(nms_threshold))


__all__ = ["YoloXAdapter", "YoloXDetection"]

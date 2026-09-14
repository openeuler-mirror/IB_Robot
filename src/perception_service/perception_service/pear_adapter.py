"""PEAR parameter-network adapter for the compiled Ascend bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from inference_service.unified_runtime import ModelResult

from .adapter_support import output_tensor, read_adapter_identity
from .perception_adapter import AdapterIdentity, PerceptionAdapter


@dataclass(frozen=True)
class PearParameters:
    smplx_pose_raw: np.ndarray
    smplx_scale: np.ndarray
    smplx_shape: np.ndarray
    smplx_expression: np.ndarray
    flame_pose: np.ndarray
    flame_shape: np.ndarray
    flame_expression: np.ndarray
    camera_raw: np.ndarray


class PearParameterAdapter(PerceptionAdapter):
    """PEAR parameter-network adapter; crops the person itself from the source frame."""

    identity = AdapterIdentity(
        "pear_parameter_network",
        "pear-rgb-crop256-bgr-imagenet-v1",
        "pear-parameter-v1",
        frozenset({"ascend_310p"}),
        operation="predict_parameters",
    )
    compiled_abi_finalized = True
    _outputs = (
        ("smplx_pose_raw", (1, 312)),
        ("smplx_scale", (1, 6)),
        ("smplx_shape", (1, 200)),
        ("smplx_expression", (1, 50)),
        ("flame_pose", (1, 14)),
        ("flame_shape", (1, 300)),
        ("flame_expression", (1, 50)),
        ("camera_raw", (1, 3)),
    )

    @classmethod
    def from_bundle(cls, bundle_root, _identity=None, *, model=None, deployment=None):
        del model, deployment
        read_adapter_identity(Path(bundle_root), cls.identity)
        return cls()

    #: Crop geometry belongs to the exported graph, not to callers. The affine target
    #: corner is ``_CROP_SIZE - 1``, not ``_CROP_SIZE``: cv2.getAffineTransform maps
    #: corner pixel centres, so using 256 here rescales every crop by 256/255 and
    #: silently shifts the regressed pose.
    _CROP_SIZE = 256
    _CROP_MARGIN = 1.25

    def preprocess(self, value: object) -> dict[str, np.ndarray]:
        image_rgb, boxes = value
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("image must be an RGB uint8 HxWx3 array")
        selected = [np.asarray(box, dtype=np.float32).reshape(4) for box in boxes]
        if len(selected) != 1:
            raise ValueError(
                f"PEAR received {len(selected)} boxes; the compiled batch-1 deployment "
                "regresses parameters for exactly one person crop per call"
            )
        crop = self._crop_person(image, selected[0])
        bgr = np.ascontiguousarray(crop[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
        return {"pear.input": bgr / np.float32(255.0)}

    @classmethod
    def _crop_person(cls, image_rgb: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
        """Square, margin-padded affine crop in source-image coordinates."""
        x1, y1, x2, y2 = (float(value) for value in box_xyxy)
        # A degenerate or non-finite box does not fail here on its own: cv2 returns a
        # uniformly padded 256x256 patch, PEAR happily regresses a pose from it, and
        # every downstream check (shape, finiteness, success) passes. Reject instead.
        if not all(np.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError(f"person bbox must be finite, got {[x1, y1, x2, y2]}")
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"person bbox must have positive extent, got {[x1, y1, x2, y2]}")
        centre_x, centre_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        half = max(x2 - x1, y2 - y1) * cls._CROP_MARGIN / 2.0
        target = float(cls._CROP_SIZE - 1)
        src = np.float32(
            [
                [centre_x - half, centre_y - half],
                [centre_x + half, centre_y - half],
                [centre_x - half, centre_y + half],
            ]
        )
        dst = np.float32([[0.0, 0.0], [target, 0.0], [0.0, target]])
        return cv2.warpAffine(
            image_rgb,
            cv2.getAffineTransform(src, dst),
            (cls._CROP_SIZE, cls._CROP_SIZE),
            borderMode=cv2.BORDER_CONSTANT,
        )

    def postprocess(self, result: ModelResult, **_options) -> PearParameters:
        values = []
        for semantic, shape in self._outputs:
            value = output_tensor(result, semantic, np.float32)
            if value.shape != shape:
                raise RuntimeError(f"PEAR output {semantic} must be float32 {shape}, got {value.shape}")
            values.append(value.reshape(-1).copy())
        return PearParameters(*values)


__all__ = ["PearParameterAdapter", "PearParameters"]

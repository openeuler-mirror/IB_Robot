"""Semantic adapters for manifest-bound perception model sessions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from inference_service.unified_runtime import ModelResult

from .adapter_support import output_tensor as _output
from .adapter_support import read_adapter_identity as _read_adapter_identity
from .grounding_dino_tokenizer import BertWordPieceTokenizer
from .model_contracts import MAX_MASK_BATCH, MAX_TEXT_BATCH
from .perception_adapter import AdapterIdentity, PerceptionAdapter

_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def _load_siglip2_tokenizer(model_path: Path):
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("SigLIP2 requires transformers for bundle-local tokenization") from exc
    return AutoTokenizer.from_pretrained(model_path, local_files_only=True)


def _normalize(rows: np.ndarray, dimension: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows[None]
    if rows.ndim != 2 or rows.shape[1] != dimension:
        raise RuntimeError(f"embedding output must have shape (batch, {dimension}), got {rows.shape}")
    if not np.isfinite(rows).all():
        raise RuntimeError("embedding output contains non-finite values")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise RuntimeError("embedding output contains a zero-norm vector")
    return np.ascontiguousarray(rows / norms, dtype=np.float32)


def _resize_rgb_batch(images: list[Image.Image], size: int) -> np.ndarray:
    rows = []
    for image in images:
        value = np.asarray(image.convert("RGB").resize((size, size), _BILINEAR), dtype=np.float32)
        rows.append(value.transpose(2, 0, 1) / np.float32(127.5) - np.float32(1.0))
    return np.ascontiguousarray(rows, dtype=np.float32)


@dataclass(frozen=True)
class SegmentationMask:
    mask: np.ndarray
    bbox_xyxy: np.ndarray
    score: float
    stability_score: float
    area: int


class SAM2Adapter(PerceptionAdapter):
    identity = AdapterIdentity(
        "sam2",
        "sam2-rgb-uint8-v1",
        "automatic-masks-v1",
        frozenset({"torch_cpu", "torch_cuda", "ascend_310p"}),
        operation="automatic",
    )
    compiled_abi_finalized = True

    @classmethod
    def from_bundle(cls, bundle_root: str | Path, _identity=None, *, model=None, deployment=None) -> SAM2Adapter:
        del deployment, model
        _read_adapter_identity(Path(bundle_root), cls.identity)
        return cls()

    def preprocess(self, image_rgb: np.ndarray) -> dict[str, np.ndarray]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError("image must be an RGB uint8 HxWx3 array")
        return {"observation.image": np.ascontiguousarray(image_rgb)}

    def postprocess(self, result: ModelResult, *, image_shape=None, **_options) -> list[SegmentationMask]:
        masks = _output(result, "masks", np.uint8)
        boxes = _output(result, "boxes", np.float32)
        scores = _output(result, "scores", np.float32).reshape(-1)
        stability = _output(result, "stability_scores", np.float32).reshape(-1)
        if (
            masks.ndim != 3
            or boxes.shape != (len(masks), 4)
            or scores.shape != (len(masks),)
            or stability.shape != (len(masks),)
        ):
            raise RuntimeError("SAM2 outputs have inconsistent batch dimensions")
        if image_shape is not None and masks.shape[1:] != tuple(image_shape):
            raise RuntimeError("SAM2 returned masks with dimensions different from the source image")
        return [
            SegmentationMask(
                mask=(masks[index] > 0).astype(np.uint8),
                bbox_xyxy=boxes[index].copy(),
                score=float(scores[index]),
                stability_score=float(stability[index]),
                area=int(np.count_nonzero(masks[index])),
            )
            for index in range(len(masks))
        ]


@dataclass(frozen=True)
class MaskEncoding:
    mask_index: int
    embedding: np.ndarray
    matched_label: str
    matched_score: float


class _SigLIP2Adapter(PerceptionAdapter):
    compiled_abi_finalized = True

    def __init__(self, dimension: int, tokenizer) -> None:
        self.dimension = dimension
        self.tokenizer = tokenizer

    @classmethod
    def from_bundle(cls, bundle_root: str | Path, identity, *, model=None, deployment=None) -> _SigLIP2Adapter:
        del deployment, model
        _read_adapter_identity(Path(bundle_root), cls.identity)
        if identity is None or identity.embedding is None:
            raise ValueError("SigLIP2 manifest semantic_identity must declare embedding metadata")
        embedding = identity.embedding
        if embedding.normalization != "l2":
            raise ValueError("SigLIP2 embedding normalization must be 'l2'")
        if not embedding.embedding_space_id:
            raise ValueError("SigLIP2 embedding_space_id must be non-empty")
        model_path = Path(bundle_root) / "assets" / "model"
        tokenizer = _load_siglip2_tokenizer(model_path)
        return cls(embedding.dimension, tokenizer)

    def _tokenize(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if not texts:
            empty = np.empty((0, 64), dtype=np.int64)
            return empty, empty.copy()
        encoded = self.tokenizer(
            [f"This is a photo of {text}." for text in texts],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="np",
        )
        tokens = np.ascontiguousarray(encoded["input_ids"], dtype=np.int64)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            pad_token_id = self.tokenizer.pad_token_id
            attention_mask = np.ones_like(tokens) if pad_token_id is None else tokens != pad_token_id
        attention = np.ascontiguousarray(attention_mask, dtype=np.int64)
        return tokens, attention


class SigLIP2ImageAdapter(_SigLIP2Adapter):
    identity = AdapterIdentity(
        "siglip2",
        "siglip2-dual-encoder-v2",
        "normalized-embedding-v1",
        frozenset({"torch_cpu", "torch_cuda", "ascend_310p", "ascend_310b"}),
        operation="encode",
    )

    def preprocess(self, value: object) -> dict[str, np.ndarray]:
        image_rgb, masks, candidate_labels = value
        if not 1 <= len(masks) <= MAX_MASK_BATCH:
            raise ValueError(f"mask batch must contain between 1 and {MAX_MASK_BATCH} masks")
        crops = []
        for index, raw_mask in enumerate(masks):
            mask = np.asarray(raw_mask) > 0
            if mask.shape != image_rgb.shape[:2]:
                raise ValueError(f"mask {index} dimensions do not match the source image")
            ys, xs = np.where(mask)
            if not len(xs):
                raise ValueError(f"mask {index} is empty")
            x1, x2, y1, y2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
            crop = image_rgb[y1:y2, x1:x2].copy()
            crop[~mask[y1:y2, x1:x2]] = 127
            crops.append(Image.fromarray(crop))
        tokens, attention = self._tokenize(list(candidate_labels))
        return {
            "masked_images": _resize_rgb_batch(crops, 384),
            "text_tokens": tokens,
            "text_attention_mask": attention,
        }

    def postprocess(self, result: ModelResult, *, candidate_labels=(), **_options) -> list[MaskEncoding]:
        image_features = _normalize(_output(result, "image_embeddings"), self.dimension)
        text_features = _normalize(_output(result, "text_embeddings"), self.dimension) if candidate_labels else None
        if text_features is not None and len(text_features) != len(candidate_labels):
            raise RuntimeError("SigLIP2 returned a different embedding count than the candidate-label batch")
        records = []
        for index, embedding in enumerate(image_features):
            label = ""
            score = 0.0
            if text_features is not None:
                similarities = text_features @ embedding
                best = int(np.argmax(similarities))
                label = candidate_labels[best]
                score = float(similarities[best])
            records.append(MaskEncoding(index, embedding, label, score))
        return records


class SigLIP2TextAdapter(_SigLIP2Adapter):
    identity = AdapterIdentity(
        "siglip2",
        "siglip2-dual-encoder-v2",
        "normalized-embedding-v1",
        frozenset({"torch_cpu", "torch_cuda", "ascend_310p", "ascend_310b"}),
        operation="encode",
    )

    def preprocess(self, texts: object) -> dict[str, np.ndarray]:
        values = list(texts)
        if not 1 <= len(values) <= MAX_TEXT_BATCH:
            raise ValueError(f"text batch must contain between 1 and {MAX_TEXT_BATCH} texts")
        tokens, attention = self._tokenize(values)
        return {
            "masked_images": np.empty((0, 3, 384, 384), dtype=np.float32),
            "text_tokens": tokens,
            "text_attention_mask": attention,
        }

    def postprocess(self, result: ModelResult, **_options) -> np.ndarray:
        return _normalize(_output(result, "text_embeddings"), self.dimension)


@dataclass(frozen=True)
class GroundingDetection:
    label: str
    confidence: float
    bbox_xyxy: np.ndarray
    mask: np.ndarray | None


class GroundingDINOAdapter(PerceptionAdapter):
    identity = AdapterIdentity(
        "grounding_dino",
        "grounded-sam2-rgb-text-thresholds-v2",
        "boxes-scores-labels-masks-v1",
        frozenset({"torch_cpu", "torch_cuda"}),
        operation="detect",
    )
    compiled_abi_finalized = False

    @classmethod
    def from_bundle(
        cls, bundle_root: str | Path, _identity=None, *, model=None, deployment=None
    ) -> GroundingDINOAdapter:
        del deployment, model
        _read_adapter_identity(Path(bundle_root), cls.identity)
        return cls()

    def preprocess(self, value: object) -> dict[str, np.ndarray]:
        image_rgb, prompt, box_threshold, text_threshold = value
        prompt_bytes = prompt.encode()
        return {
            "observation.image": np.ascontiguousarray(image_rgb, dtype=np.uint8),
            "text_prompt": np.frombuffer(prompt_bytes, dtype=np.uint8).copy(),
            "box_threshold": np.asarray([box_threshold or 0.35], dtype=np.float32),
            "text_threshold": np.asarray([text_threshold or 0.25], dtype=np.float32),
        }

    def postprocess(self, result: ModelResult, *, image_shape=None, labels=(), **_options) -> list[GroundingDetection]:
        boxes = _output(result, "boxes", np.float32)
        scores = _output(result, "scores", np.float32).reshape(-1)
        masks = _output(result, "masks", np.uint8)
        label_indices = np.asarray(result.outputs.get("label_indices"), dtype=np.int32).reshape(-1)
        if (
            boxes.shape != (len(scores), 4)
            or masks.ndim != 3
            or len(masks) != len(scores)
            or len(label_indices) != len(scores)
        ):
            raise RuntimeError("Grounding DINO outputs have inconsistent batch dimensions")
        if image_shape is not None and masks.shape[1:] != tuple(image_shape):
            raise RuntimeError("Grounding DINO returned masks with dimensions different from the source image")
        vocabulary = result.metadata.get("labels", labels)
        return [
            GroundingDetection(
                label=(
                    str(vocabulary[label_indices[index]])
                    if 0 <= label_indices[index] < len(vocabulary)
                    else str(label_indices[index])
                ),
                confidence=float(scores[index]),
                bbox_xyxy=boxes[index].copy(),
                mask=(masks[index] > 0).astype(np.uint8),
            )
            for index in range(len(scores))
        ]


def _resolve_input(deployment, model, semantic: str):
    """Find the ABI descriptor for one input semantic, preferring the compiled binding.

    `model.inputs` is deployment-agnostic, so a semantic whose batch differs per target -
    SAM2's `point_coords` is compiled at 4 on 310P and 1 on 310B - is declared there as
    `-1`. Only the selected deployment's bindings carry the shape the artifact was
    actually built for, so they win whenever the caller supplied a deployment.
    """
    for bindings in (getattr(deployment, "bindings", None) or {}).values():
        for binding in bindings.inputs:
            if binding.semantic == semantic:
                return binding
    for tensor in getattr(model, "inputs", ()) or ():
        if tensor.semantic == semantic:
            return tensor
    return None


def _input_shape(deployment, model, semantic: str, default: tuple[int, ...]) -> tuple[int, ...]:
    """Read a declared input shape from the manifest ABI, if available."""
    tensor = _resolve_input(deployment, model, semantic)
    return tuple(tensor.shape) if tensor is not None else default


def _image_geometry(
    deployment, model, semantic: str, default_shape: tuple[int, ...], default_layout: str
) -> tuple[int, int, str]:
    """Resolve ``(height, width, layout)`` for one rank-4 image input.

    The manifest declares NCHW/NHWC per rank-4 tensor, so the adapter reads the channel
    order from the ABI instead of assuming the exported graph is NCHW.
    """
    tensor = _resolve_input(deployment, model, semantic)
    shape = tuple(tensor.shape) if tensor is not None else default_shape
    layout = (tensor.layout if tensor is not None else None) or default_layout
    if len(shape) != 4:
        raise ValueError(f"image input {semantic!r} must be rank-4, got shape {shape}")
    if layout == "NCHW":
        return int(shape[2]), int(shape[3]), layout
    if layout == "NHWC":
        return int(shape[1]), int(shape[2]), layout
    raise ValueError(f"image input {semantic!r} declares unsupported layout {layout!r}")


def _arrange_image(image_hwc: np.ndarray, layout: str) -> np.ndarray:
    """Add the batch axis in the channel order the compiled graph declares."""
    ordered = image_hwc.transpose(2, 0, 1) if layout == "NCHW" else image_hwc
    return np.ascontiguousarray(ordered[None], dtype=np.float32)


class GroundingDINORawAdapter(PerceptionAdapter):
    """Manifest adapter for the 310P raw Grounding-DINO execution graph."""

    identity = AdapterIdentity(
        "grounding_dino",
        "grounding-dino-swint-rgb720x1280-bert-seq8-v1",
        "grounding-dino-raw-logits-cxcywh-v1",
        frozenset({"ascend_310p", "torch_cpu", "torch_cuda"}),
        operation="detect",
    )
    compiled_abi_finalized = True

    # Fallback ABI for callers that construct the adapter without a manifest
    # ModelDescriptor (e.g. bundle-local unit tests); real deployments derive these
    # from the compiled bindings or `model.inputs` instead (see `_resolve_input`).
    _DEFAULT_INPUT_IDS_SHAPE = (1, 8)
    _DEFAULT_ENCODER_TGT_SHAPE = (1, 900, 256)
    _DEFAULT_IMAGE_SHAPE = (1, 3, 720, 1280)
    _DEFAULT_IMAGE_LAYOUT = "NCHW"

    def __init__(self) -> None:
        self.image_height, self.image_width, self.image_layout = _image_geometry(
            None, None, "image", self._DEFAULT_IMAGE_SHAPE, self._DEFAULT_IMAGE_LAYOUT
        )

    @classmethod
    def from_bundle(
        cls, bundle_root: str | Path, _identity=None, *, model=None, deployment=None
    ) -> GroundingDINORawAdapter:
        root = Path(bundle_root)
        _read_adapter_identity(root, cls.identity)
        input_ids_shape = _input_shape(deployment, model, "input_ids", cls._DEFAULT_INPUT_IDS_SHAPE)
        sequence_length = int(input_ids_shape[-1])
        tokenizer = BertWordPieceTokenizer(
            root / "assets" / "bert-base-uncased" / "vocab.txt", sequence_length=sequence_length
        )
        target_path = root / "assets" / "encoder_tgt.npy"
        target = np.ascontiguousarray(np.load(target_path, allow_pickle=False), dtype=np.float32)
        expected_shape = _input_shape(deployment, model, "encoder_tgt", cls._DEFAULT_ENCODER_TGT_SHAPE)
        if target.shape != expected_shape:
            raise ValueError(f"Grounding-DINO encoder_tgt must have shape {expected_shape}, got {target.shape}")
        instance = cls()
        instance.tokenizer = tokenizer
        instance.encoder_target = target
        instance.image_height, instance.image_width, instance.image_layout = _image_geometry(
            deployment, model, "image", cls._DEFAULT_IMAGE_SHAPE, cls._DEFAULT_IMAGE_LAYOUT
        )
        return instance

    def preprocess(self, value: object) -> dict[str, np.ndarray]:
        image_rgb, prompt, box_threshold, text_threshold = value
        image = np.asarray(image_rgb, dtype=np.uint8)
        if image.shape[:2] != (self.image_height, self.image_width):
            image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
        normalized = image.astype(np.float32) / np.float32(255.0)
        normalized = (normalized - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        tokenized = self.tokenizer.encode(str(prompt))
        del box_threshold, text_threshold
        return {
            "image": _arrange_image(normalized, self.image_layout),
            "input_ids": tokenized["gdino.input_ids"],
            "token_type_ids": tokenized["gdino.token_type_ids"],
            "position_ids": tokenized["gdino.position_ids"],
            "text_self_attention_masks": tokenized["gdino.text_self_attention_masks"],
            "text_token_mask": tokenized["gdino.text_token_mask"],
            "encoder_tgt": self.encoder_target,
        }

    def postprocess(
        self,
        result: ModelResult,
        *,
        image_shape=None,
        prompt="",
        box_threshold=0.35,
        text_threshold=0.25,
        **_options,
    ) -> list[GroundingDetection]:
        logits = np.asarray(_output(result, "pred_logits"), dtype=np.float32)
        boxes = np.asarray(_output(result, "pred_boxes"), dtype=np.float32)
        if logits.ndim != 3 or logits.shape[0] != 1 or boxes.shape != (1, logits.shape[1], 4):
            raise RuntimeError(f"Grounding-DINO raw outputs have incompatible shapes {logits.shape} and {boxes.shape}")
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits[0], -50.0, 50.0)))
        # The "grounding-dino-raw-logits-cxcywh-v1" postprocessing contract packs one
        # classification logit per BERT prompt token into the leading
        # `sequence_length` columns of `pred_logits`; columns beyond that are unused
        # padding and must never contribute to a box's relevance score.
        prompt_token_columns = self.tokenizer.sequence_length
        if not 0 < prompt_token_columns <= logits.shape[2]:
            raise RuntimeError(
                f"Grounding-DINO {self.identity.postprocessing!r} contract expects prompt token columns "
                f"in (0, {logits.shape[2]}], got {prompt_token_columns}"
            )
        scores = probabilities[:, :prompt_token_columns].max(axis=1)
        selected = np.flatnonzero(scores > float(box_threshold or 0.35))
        height, width = tuple(image_shape or (self.image_height, self.image_width))
        prompt_input_ids = self.tokenizer.encode(str(prompt))["gdino.input_ids"]
        detections = []
        for index in selected:
            center_x, center_y, box_width, box_height = boxes[0, index]
            xyxy = np.asarray(
                [
                    (center_x - box_width / 2.0) * width,
                    (center_y - box_height / 2.0) * height,
                    (center_x + box_width / 2.0) * width,
                    (center_y + box_height / 2.0) * height,
                ],
                dtype=np.float32,
            )
            xyxy[[0, 2]] = np.clip(xyxy[[0, 2]], 0, width)
            xyxy[[1, 3]] = np.clip(xyxy[[1, 3]], 0, height)
            if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                continue
            label = (
                self.tokenizer.decode_probability_mask(
                    prompt_input_ids,
                    probabilities[index],
                    float(text_threshold or 0.25),
                )
                or str(prompt).strip()
            )
            detections.append(
                GroundingDetection(
                    label=label,
                    confidence=float(scores[index]),
                    bbox_xyxy=xyxy,
                    mask=None,
                )
            )
        return detections


class SAM2PromptAdapter(PerceptionAdapter):
    """Manifest adapter for box-prompt SAM2 execution on Ascend 310P."""

    identity = AdapterIdentity(
        "sam2",
        "sam2-longest-side1024-imagenet-box-prompt-v1",
        "sam2-mask-logits-iou-v1",
        frozenset({"ascend_310p", "ascend_310b"}),
        operation="prompt",
    )
    compiled_abi_finalized = True

    # Fallback ABI for callers that construct the adapter without a manifest
    # deployment (e.g. bundle-local unit tests). `point_coords` and `mask_input` are
    # declared with a `-1` batch on the deployment-agnostic ModelDescriptor, so the
    # real batch only ever comes from the selected deployment's bindings: 4 on 310P,
    # 1 on 310B.
    _DEFAULT_IMAGE_SHAPE = (1, 3, 1024, 1024)
    _DEFAULT_IMAGE_LAYOUT = "NCHW"
    _DEFAULT_POINT_COORDS_SHAPE = (4, 2, 2)
    _DEFAULT_MASK_INPUT_SHAPE = (4, 1, 256, 256)

    def __init__(self) -> None:
        self.batch_size = int(self._DEFAULT_POINT_COORDS_SHAPE[0])
        self.mask_input_shape = self._DEFAULT_MASK_INPUT_SHAPE
        self.image_size, _, self.image_layout = _image_geometry(
            None, None, "image", self._DEFAULT_IMAGE_SHAPE, self._DEFAULT_IMAGE_LAYOUT
        )

    @classmethod
    def from_bundle(cls, bundle_root: str | Path, _identity=None, *, model=None, deployment=None) -> SAM2PromptAdapter:
        _read_adapter_identity(Path(bundle_root), cls.identity)
        instance = cls()
        height, width, layout = _image_geometry(
            deployment, model, "image", cls._DEFAULT_IMAGE_SHAPE, cls._DEFAULT_IMAGE_LAYOUT
        )
        if height != width:
            raise ValueError(
                f"SAM2 box-prompt letterboxes onto a square canvas but the deployment declares a {height}x{width} image"
            )
        point_coords_shape = _input_shape(deployment, model, "point_coords", cls._DEFAULT_POINT_COORDS_SHAPE)
        mask_input_shape = _input_shape(deployment, model, "mask_input", cls._DEFAULT_MASK_INPUT_SHAPE)
        batch_size = int(point_coords_shape[0])
        if batch_size < 1:
            raise ValueError(
                f"SAM2 box-prompt deployment must declare a concrete point_coords batch, got shape {point_coords_shape}"
            )
        if int(mask_input_shape[0]) != batch_size:
            raise ValueError(
                f"SAM2 box-prompt deployment declares point_coords batch {batch_size} but mask_input batch "
                f"{mask_input_shape[0]}; the compiled decoder consumes one mask slot per prompt"
            )
        instance.batch_size = batch_size
        instance.mask_input_shape = mask_input_shape
        instance.image_size = height
        instance.image_layout = layout
        return instance

    def preprocess(self, value: object) -> dict[str, np.ndarray]:
        image_rgb, boxes = value
        image = np.asarray(image_rgb, dtype=np.uint8)
        height, width = image.shape[:2]
        canvas_size = int(self.image_size)
        scale = float(canvas_size) / max(height, width)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        canvas[:resized_height, :resized_width] = resized
        normalized = canvas.astype(np.float32) / np.float32(255.0)
        normalized = (normalized - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        all_boxes = [np.asarray(box, dtype=np.float32).reshape(4) for box in boxes]
        if len(all_boxes) > self.batch_size:
            raise ValueError(
                f"SAM2 box-prompt batch received {len(all_boxes)} boxes but the compiled deployment "
                f"only accepts {self.batch_size}; upstream detection filtering must cap the batch first"
            )
        selected = all_boxes
        if not selected:
            raise ValueError("at least one bounding box is required")
        coords = np.zeros((self.batch_size, 2, 2), dtype=np.float32)
        labels = np.zeros((self.batch_size, 2), dtype=np.int8)
        for index, box in enumerate(selected):
            x1, y1, x2, y2 = box
            coords[index] = np.asarray([[x1 * scale, y1 * scale], [x2 * scale, y2 * scale]], dtype=np.float32)
            labels[index] = (2, 3)
        for index in range(len(selected), self.batch_size):
            coords[index] = coords[len(selected) - 1]
            labels[index] = labels[len(selected) - 1]
        return {
            "image": _arrange_image(normalized, self.image_layout),
            "point_coords": coords,
            "point_labels": labels,
            "mask_input": np.zeros((self.batch_size, *self.mask_input_shape[1:]), dtype=np.float32),
            "has_mask_input": np.zeros((self.batch_size,), dtype=np.int8),
        }

    def postprocess(self, result: ModelResult, *, image_shape=None, count=1, **_options) -> list[np.ndarray]:
        logits = np.asarray(_output(result, "mask_logits"), dtype=np.float32)
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise RuntimeError(f"SAM2 prompt output has incompatible mask shape {logits.shape}")
        canvas_size = int(self.image_size)
        height, width = tuple(image_shape or (canvas_size, canvas_size))
        scale = float(canvas_size) / max(height, width)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        logits_height, logits_width = logits.shape[-2:]
        valid_width = min(logits_width, max(1, int(round(resized_width * logits_width / canvas_size))))
        valid_height = min(logits_height, max(1, int(round(resized_height * logits_height / canvas_size))))
        return [
            (
                cv2.resize(
                    logits[index, 0, :valid_height, :valid_width],
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )
                > 0
            ).astype(np.uint8)
            for index in range(min(int(count), len(logits)))
        ]


__all__ = [
    "GroundingDINOAdapter",
    "GroundingDINORawAdapter",
    "MaskEncoding",
    "SAM2Adapter",
    "SAM2PromptAdapter",
    "SegmentationMask",
    "SigLIP2ImageAdapter",
    "SigLIP2TextAdapter",
]

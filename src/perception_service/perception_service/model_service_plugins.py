"""Typed perception plugins backed exclusively by shared model sessions."""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np
from cv_bridge import CvBridge

from ibrobot_msgs.msg import Detection2D, DetectionArray
from inference_service.backends import BackendRegistry, RuntimeContext
from inference_service.model_service_plugin import ModelServicePlugin, PluginRuntimeStatus
from inference_service.runtime_composition import require_runtime_dependencies
from inference_service.unified_runtime import (
    ExecutionContext,
    ExecutionContract,
    ModelRequest,
    ModelRuntimeHandle,
    RegistrySet,
    RuntimeAssembly,
    RuntimeProviders,
)

from .graspgen_adapter import GraspGenAdapter
from .model_contracts import (
    point_cloud_xyz,
    rank_detections,
    validate_detection_batch,
    validate_image_message,
    validate_mask_batch,
    validate_text_batch,
)
from .pear_adapter import PearParameterAdapter
from .ram_plus_adapter import RAMPlusAdapter, masked_image_crop, select_mask_tags
from .semantic_model_adapters import (
    GroundingDINOAdapter,
    GroundingDINORawAdapter,
    SAM2Adapter,
    SAM2PromptAdapter,
    SigLIP2ImageAdapter,
    SigLIP2TextAdapter,
)
from .yolox_adapter import YoloXAdapter


def _detection_array(bridge, header, records) -> DetectionArray:
    detections = []
    for record in records:
        detection = Detection2D()
        detection.header = header
        detection.confidence = float(getattr(record, "score", getattr(record, "confidence", 0.0)))
        detection.bbox = np.asarray(record.bbox_xyxy, dtype=float).tolist()
        if record.mask is not None:
            detection.mask = bridge.cv2_to_imgmsg((record.mask > 0).astype(np.uint8) * 255, encoding="mono8")
            detection.mask.header = header
        detection.label = str(getattr(record, "label", ""))
        detections.append(detection)
    return DetectionArray(header=header, detections=detections)


def _filter_masks(records, image_area: int, max_masks: int, min_pixels: int, min_area_ratio: float, max_overlap: float):
    candidates = [
        (index, record)
        for index, record in enumerate(records)
        if record.area >= min_pixels and record.area / image_area >= min_area_ratio
    ]
    candidates.sort(key=lambda item: (-item[1].score, -item[1].area, item[0]))
    accepted = []
    for _, record in candidates:
        mask = record.mask > 0
        if any(np.count_nonzero(mask & (other.mask > 0)) / max(1, record.area) > max_overlap for other in accepted):
            continue
        accepted.append(record)
        if max_masks and len(accepted) >= max_masks:
            break
    return accepted


class _SessionPlugin(ModelServicePlugin):
    service_type = ""
    model_type = ""
    # ``operation`` is the v3 service contract.  Grounding-DINO deployment
    # variants share ``detect`` and remain an adapter/deployment detail.
    operation = ""
    execution_structure = "direct"
    adapter_class = None
    _session_factory: Callable | None = None
    _registry: BackendRegistry | None = None

    def __init__(
        self,
        host,
        validated,
        options,
        *,
        registry_set: RegistrySet | None = None,
        providers: RuntimeProviders | None = None,
    ) -> None:
        registry_set, providers = require_runtime_dependencies(
            registry_set,
            providers,
            owner=f"{type(self).__name__}",
        )
        model = validated.manifest.model
        expected_model_type = self.model_type
        if (
            model.interface != "tensor_model"
            or model.model_type != expected_model_type
            or model.operation != self.operation
        ):
            raise ValueError(
                f"plugin requires tensor_model/{expected_model_type}/{self.operation}, "
                f"got {model.interface}/{model.model_type}/{model.operation}"
            )
        if any(name in options for name in ("backend", "device", "model_backend", "fallback")):
            raise ValueError(
                "plugins accept only a validated named deployment; raw backend/device/fallback is forbidden"
            )
        self.host = host
        self.bridge = CvBridge()
        self.validated = validated
        self.adapter = self.adapter_class.from_bundle(
            validated.bundle_root, model.semantic_identity, model=model, deployment=validated.deployment
        )
        self.adapter.validate_identity(model.semantic_identity)
        self._closed = False
        self._requests = itertools.count(1)
        runtime_profile = getattr(validated, "runtime_profile", None)
        if runtime_profile is None:
            role_profiles = getattr(validated, "role_runtime_profiles", {})
            runtime_profile = next(iter(role_profiles.values()), None)
        context = RuntimeContext(
            validated_manifest=validated,
            runtime_options=options,
            runtime_profile=runtime_profile,
        )
        override = None
        if self._session_factory is not None:

            def override(_context, **_kwargs):
                return self._session_factory(self.model_type, self.adapter, validated, options)

        self._registry = self._registry or registry_set.backend_registry
        self.session = registry_set.session_builder_registry.create(
            context,
            allowed_deployments=self.adapter.identity.supported_deployments,
            backend_registry=self._registry,
            providers=providers,
            override=override,
            builder_options={"adapter": self.adapter},
        )
        contract = ExecutionContract(
            execution_structure=self.execution_structure,
            orchestration_visibility=("session" if self.execution_structure == "iterative" else None),
            cancellation_granularity="checkpoint" if self.execution_structure == "iterative" else "request_boundary",
        )
        self._runtime_handle = ModelRuntimeHandle(
            RuntimeAssembly(
                runtime_executor=self.session,
                session=self.session,
                execution_contract=contract,
                stateful=bool(self.session.capabilities.stateful),
                resettable=bool(self.session.capabilities.resettable),
                runtime_id=f"perception-{self.model_type}-{self.operation or 'default'}",
                load_context=context,
                providers=providers,
            )
        )
        self.pipeline = self._runtime_handle
        try:
            self._runtime_handle.load(context)
        except Exception:
            self._closed = True
            raise

    def _infer(self, inputs):
        request_id = f"{self.model_type}-{next(self._requests)}"
        return self._runtime_handle.execute(
            ModelRequest(inputs, metadata={"service_type": self.service_type}),
            ExecutionContext(request_id),
        )

    def image_rgb(self, image):
        validate_image_message(image)
        return np.asarray(self.bridge.imgmsg_to_cv2(image, desired_encoding="rgb8"), dtype=np.uint8)

    def runtime_status(self) -> PluginRuntimeStatus:
        diagnostics = self._runtime_handle.diagnostics()
        health = diagnostics.health
        state = diagnostics.state.value
        runtime_version = self.session.runtime_version
        ready = health.ready and bool(runtime_version)
        return PluginRuntimeStatus(
            state=state,
            ready=ready,
            failure_reason=""
            if ready
            else (
                "loaded model runtime did not expose a version"
                if health.ready
                else (health.message or health.reason_code or f"model runtime is {state}")
            ),
            metadata={"runtime_version": runtime_version} if runtime_version else {},
        )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._runtime_handle.close()


class RAMPlusRecognizeTagsPlugin(_SessionPlugin):
    service_type = "ibrobot_msgs/srv/RecognizeTags"
    model_type = "ram_plus"
    operation = "recognize_tags"
    adapter_class = RAMPlusAdapter

    def handle(self, request, response) -> str:
        image = self.image_rgb(request.image)
        if request.masks:
            validate_mask_batch(request.image, request.masks)
        masks = [self.bridge.imgmsg_to_cv2(mask, desired_encoding="mono8") > 0 for mask in request.masks]
        crops = [masked_image_crop(image, mask) for mask in masks]
        tags = []
        include_image = bool(request.include_image or not crops)
        if include_image:
            result = self._infer(self.adapter.preprocess(image))
            tags = self.adapter.postprocess(result, score_threshold=float(request.score_threshold))
        mask_tags = []
        mask_scores = []
        mask_tag_counts = []
        if crops:
            batch_result = self.adapter.preprocess_batch(crops)
            batch = self._infer(batch_result)
            per_mask = self.adapter.postprocess_batch(batch, score_threshold=float(request.score_threshold))
            for values in per_mask:
                candidates = select_mask_tags(
                    values,
                    excluded_labels=request.excluded_labels,
                    limit=int(request.max_mask_candidates),
                )
                mask_tag_counts.append(len(candidates))
                mask_tags.extend(value.label for value in candidates)
                mask_scores.extend(value.score for value in candidates)
        response.tags = [tag.label for tag in tags]
        response.scores = [tag.score for tag in tags]
        response.mask_tag_counts = mask_tag_counts
        response.mask_tags = mask_tags
        response.mask_scores = mask_scores
        return f"recognized {len(tags)} image tags and {len(mask_tags)} mask candidates"


class SAM2GenerateMasksPlugin(_SessionPlugin):
    service_type = "ibrobot_msgs/srv/GenerateMasks"
    model_type = "sam2"
    operation = "automatic"
    adapter_class = SAM2Adapter

    def handle(self, request, response) -> str:
        image = self.image_rgb(request.image)
        result = self._infer(self.adapter.preprocess(image))
        records = _filter_masks(
            self.adapter.postprocess(result, image_shape=image.shape[:2]),
            image.shape[0] * image.shape[1],
            int(request.max_masks),
            int(request.min_mask_pixels),
            max(0.0, float(request.min_mask_area_ratio)),
            max(0.0, float(request.max_overlap_ratio)),
        )
        response.detections = _detection_array(self.bridge, request.image.header, records)
        return f"generated {len(records)} masks"


class SigLIP2EncodeEmbeddingsPlugin(_SessionPlugin):
    service_type = "ibrobot_msgs/srv/EncodeEmbeddings"
    model_type = "siglip2"
    operation = "encode"
    adapter_class = SigLIP2ImageAdapter

    def handle(self, request, response) -> str:
        from ibrobot_msgs.msg import MaskEmbedding

        if len(request.candidate_labels) > 16:
            raise ValueError("candidate-label batch exceeds limit 16")
        if any(not label.strip() for label in request.candidate_labels):
            raise ValueError("candidate labels must not be empty")
        if not request.masks:
            validate_image_message(request.image)
            response.results = []
            return "encoded 0 masks"
        validate_mask_batch(request.image, request.masks)
        image = self.image_rgb(request.image)
        masks = [self.bridge.imgmsg_to_cv2(mask, desired_encoding="mono8") for mask in request.masks]
        labels = list(request.candidate_labels)
        result = self._infer(self.adapter.preprocess((image, masks, labels)))
        records = self.adapter.postprocess(result, candidate_labels=labels)
        if len(records) != len(masks):
            raise RuntimeError("SigLIP2 returned a different embedding count than the mask batch")
        response.results = [
            MaskEmbedding(
                mask_index=record.mask_index,
                embedding=record.embedding.astype(float).tolist(),
                embedding_dim=len(record.embedding),
                matched_label=record.matched_label,
                matched_score=record.matched_score,
                success=True,
            )
            for record in records
        ]
        return f"encoded {len(records)} masks"


class SigLIP2EncodeTextPlugin(_SessionPlugin):
    service_type = "ibrobot_msgs/srv/EncodeText"
    model_type = "siglip2"
    operation = "encode"
    adapter_class = SigLIP2TextAdapter

    def handle(self, request, response) -> str:
        from ibrobot_msgs.msg import TextEmbedding

        validate_text_batch(request.texts)
        result = self._infer(self.adapter.preprocess(request.texts))
        features = self.adapter.postprocess(result)
        if len(features) != len(request.texts):
            raise RuntimeError("SigLIP2 returned a different embedding count than the text batch")
        response.results = [
            TextEmbedding(
                text_index=index,
                embedding=embedding.astype(float).tolist(),
                embedding_dim=len(embedding),
                success=True,
            )
            for index, embedding in enumerate(features)
        ]
        return f"encoded {len(features)} texts"


class GroundingDetectPlugin(_SessionPlugin):
    service_type = "ibrobot_msgs/srv/GroundingDetect"
    model_type = "grounding_dino"
    operation = "detect"
    adapter_class = GroundingDINOAdapter

    def handle(self, request, response) -> str:
        if not request.text_prompt.strip():
            raise ValueError("text prompt must not be empty")
        image = self.image_rgb(request.image)
        result = self._infer(
            self.adapter.preprocess(
                (image, request.text_prompt, float(request.box_threshold), float(request.text_threshold))
            )
        )
        records = rank_detections(
            [
                record
                for record in self.adapter.postprocess(
                    result,
                    image_shape=image.shape[:2],
                    labels=(request.text_prompt,),
                )
                if record.confidence >= (float(request.box_threshold) or 0.35)
            ]
        )
        response.detections = _detection_array(self.bridge, request.image.header, records)
        return f"confirmed {len(records)} detections"


class YoloXPersonDetectPlugin(_SessionPlugin):
    service_type = "ibrobot_msgs/srv/YoloXDetect"
    model_type = "yolox_person"
    operation = "detect"
    adapter_class = YoloXAdapter

    def handle(self, request, response) -> str:
        confidence_threshold = float(request.confidence_threshold)
        nms_threshold = float(request.nms_threshold)
        if not np.isfinite(confidence_threshold) or not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be finite and within [0, 1]")
        if not np.isfinite(nms_threshold) or not 0.0 <= nms_threshold <= 1.0:
            raise ValueError("nms_threshold must be finite and within [0, 1]")
        image = self.image_rgb(request.image)
        result = self._infer(self.adapter.preprocess(image))
        records = rank_detections(
            [
                record
                for record in self.adapter.postprocess(result, image_shape=image.shape[:2], nms_threshold=nms_threshold)
                if record.confidence >= confidence_threshold
            ]
        )
        response.detections = _detection_array(self.bridge, request.image.header, records)
        return f"detected {len(records)} persons"


class PearParameterPredictPlugin(_SessionPlugin):
    service_type = "ibrobot_msgs/srv/PearParameterPredict"
    model_type = "pear_parameter_network"
    operation = "predict_parameters"
    adapter_class = PearParameterAdapter

    def handle(self, request, response) -> str:
        detections = list(request.detections.detections)
        validate_detection_batch(detections)
        image = self.image_rgb(request.image)
        result = self._infer(self.adapter.preprocess((image, [detection.bbox for detection in detections])))
        values = self.adapter.postprocess(result)
        response.header = request.image.header
        response.smplx_pose_raw = values.smplx_pose_raw.tolist()
        response.smplx_scale = values.smplx_scale.tolist()
        response.smplx_shape = values.smplx_shape.tolist()
        response.smplx_expression = values.smplx_expression.tolist()
        response.flame_pose = values.flame_pose.tolist()
        response.flame_shape = values.flame_shape.tolist()
        response.flame_expression = values.flame_expression.tolist()
        response.camera_raw = values.camera_raw.tolist()
        return "predicted PEAR parameters"


class GroundingDINORawDetectPlugin(_SessionPlugin):
    """Text detection service for a compiled raw Grounding-DINO deployment."""

    service_type = "ibrobot_msgs/srv/GroundingDetect"
    model_type = "grounding_dino"
    operation = "detect"
    adapter_class = GroundingDINORawAdapter

    def handle(self, request, response) -> str:
        if not request.text_prompt.strip():
            raise ValueError("text prompt must not be empty")
        image = self.image_rgb(request.image)
        result = self._infer(
            self.adapter.preprocess(
                (image, request.text_prompt, float(request.box_threshold), float(request.text_threshold))
            )
        )
        records = rank_detections(
            self.adapter.postprocess(
                result,
                image_shape=image.shape[:2],
                prompt=request.text_prompt,
                box_threshold=float(request.box_threshold),
                text_threshold=float(request.text_threshold),
            )
        )
        response.detections = _detection_array(self.bridge, request.image.header, records)
        return f"detected {len(records)} boxes"


class SegmentDetectionsPlugin(_SessionPlugin):
    """Fill masks for detections using a manifest-bound SAM2 box-prompt service."""

    service_type = "ibrobot_msgs/srv/SegmentDetections"
    model_type = "sam2"
    operation = "prompt"
    adapter_class = SAM2PromptAdapter

    def handle(self, request, response) -> str:
        source = list(request.detections.detections)
        validate_detection_batch(source)
        if not source:
            response.detections = DetectionArray(header=request.image.header, detections=[])
            return "segmented 0 detections"
        image = self.image_rgb(request.image)
        masks = self._segment_in_manifest_batches(image, source)
        output = []
        for detection, mask in zip(source, masks, strict=True):
            record = Detection2D()
            record.header = request.image.header
            record.label = detection.label
            record.confidence = detection.confidence
            record.bbox = detection.bbox
            record.mask = self.bridge.cv2_to_imgmsg((mask > 0).astype(np.uint8) * 255, encoding="mono8")
            record.mask.header = request.image.header
            output.append(record)
        response.detections = DetectionArray(header=request.image.header, detections=output)
        return f"segmented {len(output)} detections"

    def _segment_in_manifest_batches(self, image, source) -> list[np.ndarray]:
        """Segment every detection, one compiled decoder batch at a time.

        The compiled SAM2 decoder accepts a fixed number of box prompts while
        `GroundingDetect` may legitimately confirm up to `MAX_DETECTIONS` objects, so
        the service slices the request into manifest-sized chunks and concatenates the
        masks back in the original detection order instead of truncating the tail.
        """
        batch_size = int(getattr(self.adapter, "batch_size", 0)) or len(source)
        masks: list[np.ndarray] = []
        for start in range(0, len(source), batch_size):
            chunk = source[start : start + batch_size]
            result = self._infer(self.adapter.preprocess((image, [detection.bbox for detection in chunk])))
            chunk_masks = self.adapter.postprocess(result, image_shape=image.shape[:2], count=len(chunk))
            if len(chunk_masks) != len(chunk):
                raise RuntimeError(
                    f"SAM2 box-prompt deployment returned {len(chunk_masks)} masks for {len(chunk)} detections"
                )
            masks.extend(chunk_masks)
        return masks


class GraspGenGenerateGraspsPlugin(_SessionPlugin):
    service_type = "ibrobot_msgs/srv/GenerateGrasps"
    model_type = "graspgen"
    operation = "generate_grasps"
    execution_structure = "iterative"
    adapter_class = GraspGenAdapter

    def handle(self, request, response) -> str:
        from ibrobot_msgs.msg import GraspCandidate, GraspCandidateArray

        inputs, object_center = self.adapter.prepare(point_cloud_xyz(request.object_points))
        result = self._infer(inputs)
        candidates = self.adapter.postprocess(
            result,
            object_center=object_center,
            max_grasps=int(request.max_grasps),
            min_confidence=float(request.min_confidence),
        )
        header = request.object_points.header
        # Width and collision fields stay at their defaults: GraspGen scores a pose, it
        # does not measure the gripper aperture or clear the scene. manipulation_execution
        # fills those in from its own geometry pass.
        response.grasps = GraspCandidateArray(
            header=header,
            grasps=[
                GraspCandidate(
                    header=header,
                    pose_matrix=candidate.pose_matrix.reshape(16).astype(float).tolist(),
                    confidence=candidate.confidence,
                )
                for candidate in candidates
            ],
        )
        return f"generated {len(candidates)} grasps"


__all__ = [
    "GraspGenGenerateGraspsPlugin",
    "GroundingDetectPlugin",
    "GroundingDINORawDetectPlugin",
    "RAMPlusRecognizeTagsPlugin",
    "SAM2GenerateMasksPlugin",
    "SegmentDetectionsPlugin",
    "SigLIP2EncodeEmbeddingsPlugin",
    "SigLIP2EncodeTextPlugin",
]

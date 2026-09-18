"""Policy facade over the unified model runtime.

``InferencePipeline`` preserves the existing policy request/result contract
while routing all local execution through ``ModelRuntimeHandle``.
"""

from __future__ import annotations

import copy
import json
import math
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np

from ibrobot_tracing import get_trace_emitter
from inference_manifest import CompiledDeployment
from inference_service.backends import (
    BackendAdmissionError,
    BackendHealth,
    BackendState,
    InferenceRequest,
    RuntimeContext,
)
from inference_service.codecs import CodecRequest, CodecResult, ExecutionPlan, PolicyCodec, build_execution_plan
from inference_service.pi05_schedule import PI05DenoisingSchedule
from inference_service.pipeline.errors import (
    PipelineCanceledError,
    PipelineConfigurationError,
    PipelineLifecycleError,
    PipelineNotReadyError,
    PipelineTimeoutError,
    PipelineValidationError,
)
from inference_service.pipeline.executor import SequentialModelExecutor
from inference_service.pipeline.runtime_core import (
    ExecutionError,
    ModelExecutor,
    StageFrame,
)
from inference_service.pipeline.staged_executor import StagedModelExecutor, StagedScheduling
from inference_service.pipeline.state import PipelineState
from inference_service.pipeline.types import PipelineDiagnostics, PipelineResult
from inference_service.pipeline.validation import validate_action_output
from inference_service.unified_runtime import (
    Deadline,
    ExecutionContext,
    ExecutionContract,
    ExecutionFailure,
    LifecycleState,
    ModelRequest,
    ModelResult,
    ModelRuntimeHandle,
    OwnedComponent,
    RuntimeAssembly,
)

Processor = Callable[[Mapping[str, object]], Mapping[str, object]]
Postprocessor = Callable[[object], object]

trace = get_trace_emitter("ib_trace.policy", component_id="inference_pipeline")
_model_span: ContextVar[object | None] = ContextVar("policy_model_span", default=None)


def _identity_preprocessor(inputs: Mapping[str, object]) -> Mapping[str, object]:
    return inputs


def _identity_postprocessor(action: object) -> object:
    return action


def _snapshot_action(action: object) -> object:
    detached = getattr(action, "detach", None)
    candidate = detached() if callable(detached) else action
    clone = getattr(candidate, "clone", None)
    if callable(clone):
        return clone()
    return copy.deepcopy(candidate)


def _chunk_size_from_action(action: np.ndarray) -> int:
    shape = getattr(action, "shape", ())
    if len(shape) < 2 or shape[-2] < 1:
        raise PipelineValidationError(
            f"model session action output has invalid shape {shape}",
            pipeline_id="policy",
            code="invalid_action_shape",
        )
    return int(shape[-2])


@dataclass(frozen=True)
class _PolicyFrameResult:
    """Internal policy frame used before the facade publishes ``PipelineResult``.

    This is intentionally not a second public result contract.  It exists only
    because legacy policy backends and compiled session stages expose action
    metadata at different points during the transition.
    """

    action: object
    actual_chunk_size: int
    backend_latency_ms: float
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.actual_chunk_size < 1:
            raise ValueError("policy frame actual_chunk_size must be positive")
        if not math.isfinite(self.backend_latency_ms) or self.backend_latency_ms < 0:
            raise ValueError("policy frame backend latency must be finite and non-negative")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _normalize_policy_frame(result: object, *, pipeline_id: str) -> _PolicyFrameResult:
    """Normalize legacy/backend or unified session output into an internal frame."""

    if isinstance(result, _PolicyFrameResult):
        return result
    if isinstance(result, ModelResult):
        outputs = result.outputs
        action = outputs.get("action", outputs.get("actions", outputs)) if isinstance(outputs, Mapping) else outputs
        metadata = dict(result.metadata)
        chunk_size = metadata.get("actual_chunk_size")
        if type(chunk_size) is not int or chunk_size < 1:
            chunk_size = _chunk_size_from_action(action)
        return _PolicyFrameResult(action, chunk_size, result.latency_ms, metadata)

    action = getattr(result, "action", None)
    if action is None and isinstance(result, Mapping):
        action = result.get("action", result.get("outputs"))
    if action is None:
        raise PipelineValidationError(
            f"pipeline {pipeline_id!r} returned an unsupported policy result {type(result).__name__}",
            pipeline_id=pipeline_id,
            code="invalid_policy_result",
        )
    actual_chunk_size = getattr(result, "actual_chunk_size", None)
    if actual_chunk_size is None and isinstance(result, Mapping):
        actual_chunk_size = result.get("actual_chunk_size")
    if type(actual_chunk_size) is not int or actual_chunk_size < 1:
        raise PipelineValidationError(
            f"pipeline {pipeline_id!r} returned an invalid actual_chunk_size",
            pipeline_id=pipeline_id,
            code="invalid_policy_result",
        )
    backend_latency_ms = getattr(result, "backend_latency_ms", 0.0)
    if isinstance(result, Mapping):
        backend_latency_ms = result.get("backend_latency_ms", backend_latency_ms)
    metadata = getattr(result, "metadata", {})
    if isinstance(result, Mapping):
        metadata = result.get("metadata", metadata)
    if not isinstance(metadata, Mapping):
        metadata = {}
    return _PolicyFrameResult(action, actual_chunk_size, float(backend_latency_ms), metadata)


def _policy_request(frame: StageFrame) -> ModelRequest:
    request = frame.request
    if not isinstance(request, ModelRequest):
        raise PipelineValidationError(
            f"policy stage requires ModelRequest, got {type(request).__name__}",
            pipeline_id="policy",
            code="invalid_model_request",
        )
    return request


class _PolicyPreprocessStage:
    """Policy preprocess: prompt selection, processor, control-input merge, deadline gate."""

    def __init__(self, facade: InferencePipeline) -> None:
        self._facade = facade

    def execute(self, frame: StageFrame, *, deadline: datetime | None) -> None:
        request = _policy_request(frame)
        frame.values["_total_start"] = time.perf_counter()
        reserved = {
            _CONTROL_INPUTS_KEY,
            _CAPTURE_RAW_ACTION_KEY,
            _PROMPT_KEY,
            _PRIORITY_KEY,
            "_execution_context",
        }
        control_inputs = request.inputs.get(_CONTROL_INPUTS_KEY)
        if control_inputs is not None and not isinstance(control_inputs, Mapping):
            raise PipelineValidationError(
                "policy control inputs must be a mapping",
                pipeline_id=self._facade.pipeline_id,
                code="invalid_control_inputs",
            )
        prompt_value = request.inputs.get(_PROMPT_KEY)
        selected_prompt = self._facade._default_task if prompt_value is None else prompt_value
        processor_inputs = {key: value for key, value in request.inputs.items() if key not in reserved}
        if selected_prompt is not None:
            processor_inputs["task"] = selected_prompt

        frame.control.raise_if_canceled("preprocess")

        if trace.enabled and self._facade._instrument_processors and self._facade._stage_policy == "sequential":
            trace.flow_receive(
                "observation_to_preprocess",
                frame.control.request_id,
                component_id="policy.preprocess",
                pipeline_id=self._facade.pipeline_id,
            )
        preprocess_span = (
            trace.span(
                "preprocess",
                component_id="policy.preprocess",
                origin="built-in",
                pipeline_id=self._facade.pipeline_id,
            )
            if trace.enabled and self._facade._instrument_processors
            else nullcontext()
        )
        with preprocess_span:
            preprocess_start = time.perf_counter()
            canonical_inputs = self._facade._preprocessor(processor_inputs)
            preprocess_latency_ms = (time.perf_counter() - preprocess_start) * 1000.0
        if not isinstance(canonical_inputs, Mapping):
            raise PipelineValidationError(
                f"pipeline {self._facade.pipeline_id!r} preprocessor must return a mapping",
                pipeline_id=self._facade.pipeline_id,
                details={"returned_type": type(canonical_inputs).__name__},
            )
        canonical_inputs = dict(canonical_inputs)
        if control_inputs:
            collisions = sorted(set(canonical_inputs) & set(control_inputs))
            if collisions:
                raise PipelineValidationError(
                    f"pipeline {self._facade.pipeline_id!r} control inputs conflict with preprocessor "
                    f"outputs: {collisions}",
                    pipeline_id=self._facade.pipeline_id,
                    details={"conflicting_inputs": tuple(collisions)},
                )
            canonical_inputs.update(control_inputs)
        self._facade._raise_if_expired(deadline, phase="preprocess", backend_completed=False)

        frame.values["_canonical_inputs"] = canonical_inputs
        frame.values["_selected_prompt"] = selected_prompt
        frame.values["_capture_raw_action"] = bool(request.inputs.get(_CAPTURE_RAW_ACTION_KEY, False))
        frame.values["_preprocess_latency_ms"] = preprocess_latency_ms


class _RawActionResultAdapter:
    """Expose the iterative final action semantic as the executor result."""

    def __init__(self, action_semantic: str = "action") -> None:
        self._action_semantic = action_semantic

    def adapt(self, frame: StageFrame) -> object:
        return frame.values[self._action_semantic]

    def adapt_error(self, error: ExecutionError) -> object:
        if error.cause is not None:
            raise error.cause
        raise RuntimeError(error.message)


class _PolicyModelRequestStage:
    """Bind canonical policy values to the model executor request contract."""

    def __init__(
        self,
        facade: InferencePipeline,
    ) -> None:
        self._facade = facade

    def execute(self, frame: StageFrame, *, deadline: datetime | None) -> None:
        frame.control.raise_if_canceled("model.request")
        canonical_inputs = frame.values["_canonical_inputs"]
        if self._facade._execution_plan is not None:
            role_inputs = self._facade._encode_role_inputs(canonical_inputs)
            semantic_values: dict[str, object] = {}
            for bound_inputs in role_inputs.values():
                for tensor in bound_inputs.tensors:
                    semantic_values[tensor.semantic] = tensor.value
        else:
            semantic_values = dict(canonical_inputs)
        request = _policy_request(frame)
        model_request = ModelRequest(
            MappingProxyType(semantic_values),
            {**request.metadata, "priority": request.inputs.get(_PRIORITY_KEY, 0)},
        )
        frame.values["_model_request"] = model_request
        frame.values["_model_inputs"] = semantic_values
        frame.values["_role_inputs"] = role_inputs if self._facade._execution_plan is not None else None
        frame.values.update(semantic_values)
        if trace.enabled and self._facade._stage_policy == "sequential":
            if self._facade._instrument_processors:
                trace.flow_send(
                    "preprocess_to_inference",
                    frame.control.request_id,
                    component_id="policy.preprocess",
                    pipeline_id=self._facade.pipeline_id,
                )
                trace.flow_receive(
                    "preprocess_to_inference",
                    frame.control.request_id,
                    component_id=self._facade._model_component_id,
                    pipeline_id=self._facade.pipeline_id,
                )
            _model_span.set(
                trace.start_span(
                    "model_call",
                    component_id=self._facade._model_component_id,
                    origin="built-in",
                    pipeline_id=self._facade.pipeline_id,
                )
            )
        frame.values["_backend_start"] = time.perf_counter()


class _PolicyModelResultStage:
    """Wrap the flattened model stages' action in the policy backend contract."""

    def __init__(
        self,
        facade: InferencePipeline,
        action_semantic: str,
        denoising_schedule_metadata: Mapping[str, object] | None,
        metadata_provider=None,
    ) -> None:
        self._facade = facade
        self._action_semantic = action_semantic
        self._denoising_schedule_metadata = denoising_schedule_metadata
        self._metadata_provider = metadata_provider

    def execute(self, frame: StageFrame, *, deadline: datetime | None) -> None:
        raw_action = frame.values[self._action_semantic]
        if not hasattr(raw_action, "shape"):
            raise PipelineValidationError(
                f"pipeline {self._facade.pipeline_id!r} model session returned "
                f"{type(raw_action).__name__}, expected a tensor-like action",
                pipeline_id=self._facade.pipeline_id,
            )
        self._facade._raise_if_expired(deadline, phase="backend", backend_completed=True)
        self._facade._ensure_session_ready_after_call()
        request = _policy_request(frame)
        metadata = {
            "request_id": frame.control.request_id,
            "deployment_name": self._facade._context.deployment_name,
            "deployment_fingerprint": self._facade._context.deployment_fingerprint,
        }
        if callable(self._metadata_provider):
            metadata.update(self._metadata_provider(frame.control.request_id))
        chunk_size = 1 if metadata.get("action_method") == "select_action" else _chunk_size_from_action(raw_action)
        priority_mapping = self._facade.capabilities.priority_mapping
        if priority_mapping is not None:
            metadata["hardware_priority"] = priority_mapping.map_generic(int(request.inputs.get(_PRIORITY_KEY, 0)))
        if self._denoising_schedule_metadata is not None:
            metadata["denoising_schedule"] = self._denoising_schedule_metadata
        backend_start = frame.values.get("_backend_start")
        backend_latency_ms = 0.0 if backend_start is None else (time.perf_counter() - backend_start) * 1000.0
        frame.values["_backend_result"] = _PolicyFrameResult(
            action=raw_action,
            actual_chunk_size=chunk_size,
            backend_latency_ms=backend_latency_ms,
            metadata=metadata,
        )
        if trace.enabled and self._facade._stage_policy == "sequential":
            trace.end_span(_model_span.get())
            _model_span.set(None)
            if self._facade._instrument_processors:
                trace.flow_send(
                    "inference_to_postprocess",
                    frame.control.request_id,
                    component_id=self._facade._model_component_id,
                    pipeline_id=self._facade.pipeline_id,
                )


class _PolicyCompletionStage:
    def __init__(self, operation: Callable[[], None]) -> None:
        self._operation = operation

    def execute(self, frame: StageFrame, *, deadline: datetime | None) -> None:
        del frame, deadline
        self._operation()


class _PolicyDecodeStage:
    """Decode/validate/raw-capture stage: action decode, validation, optional raw snapshot."""

    def __init__(self, facade: InferencePipeline) -> None:
        self._facade = facade

    def execute(self, frame: StageFrame, *, deadline: datetime | None) -> None:
        del deadline
        frame.control.raise_if_canceled("decode")
        backend_result = frame.values["_backend_result"]

        semantic_action = self._facade._decode_backend_action(backend_result)
        validate_action_output(
            semantic_action,
            actual_chunk_size=backend_result.actual_chunk_size,
            action_dimension=self._facade._action_dimension,
            pipeline_id=self._facade.pipeline_id,
            phase="backend",
        )
        raw_action = _snapshot_action(semantic_action) if frame.values.get("_capture_raw_action", False) else None

        frame.values["_semantic_action"] = semantic_action
        frame.values["_raw_action"] = raw_action


class _PolicyPostprocessStage:
    """Postprocess/validate stage: postprocessor, validation, postprocess deadline gate."""

    def __init__(self, facade: InferencePipeline) -> None:
        self._facade = facade

    def execute(self, frame: StageFrame, *, deadline: datetime | None) -> None:
        frame.control.raise_if_canceled("postprocess")
        backend_result = frame.values["_backend_result"]

        if trace.enabled and self._facade._instrument_processors and self._facade._stage_policy == "sequential":
            trace.flow_receive(
                "inference_to_postprocess",
                frame.control.request_id,
                component_id="policy.postprocess",
                pipeline_id=self._facade.pipeline_id,
            )
        postprocess_span = (
            trace.span(
                "postprocess",
                component_id="policy.postprocess",
                origin="built-in",
                pipeline_id=self._facade.pipeline_id,
            )
            if trace.enabled and self._facade._instrument_processors
            else nullcontext()
        )
        with postprocess_span:
            postprocess_start = time.perf_counter()
            action = self._facade._postprocessor(frame.values["_semantic_action"])
            postprocess_latency_ms = (time.perf_counter() - postprocess_start) * 1000.0
        validate_action_output(
            action,
            actual_chunk_size=backend_result.actual_chunk_size,
            action_dimension=self._facade._action_dimension,
            pipeline_id=self._facade.pipeline_id,
            phase="postprocessor",
        )
        self._facade._raise_if_expired(deadline, phase="postprocess", backend_completed=True)

        frame.values["_action"] = action
        frame.values["_postprocess_latency_ms"] = postprocess_latency_ms


class _PolicyResultAdapter:
    """Result adapter: assemble the policy PipelineResult from the completed stage frame."""

    def __init__(self, facade: InferencePipeline) -> None:
        self._facade = facade

    def adapt(self, frame: StageFrame) -> PipelineResult:
        backend_result = frame.values["_backend_result"]
        total_latency_ms = (time.perf_counter() - frame.values["_total_start"]) * 1000.0
        preprocess_latency_ms = frame.values["_preprocess_latency_ms"]
        postprocess_latency_ms = frame.values["_postprocess_latency_ms"]
        latency_metadata = MappingProxyType(
            {
                "total": total_latency_ms,
                "preprocess": preprocess_latency_ms,
                "backend": backend_result.backend_latency_ms,
                "postprocess": postprocess_latency_ms,
            }
        )
        manifest = self._facade._context.validated_manifest.manifest
        deployment = self._facade._context.deployment
        return PipelineResult(
            action=frame.values["_action"],
            actual_chunk_size=backend_result.actual_chunk_size,
            pipeline_id=self._facade.pipeline_id,
            bundle=manifest.bundle.name,
            bundle_uuid=manifest.bundle.uuid,
            bundle_revision=manifest.bundle.revision,
            deployment=self._facade._context.deployment_name,
            deployment_uuid=deployment.uuid,
            deployment_revision=deployment.revision,
            deployment_fingerprint=self._facade._context.deployment_fingerprint,
            backend=self._facade._backend_name(),
            state=self._facade.state,
            total_latency_ms=total_latency_ms,
            preprocess_latency_ms=preprocess_latency_ms,
            backend_latency_ms=backend_result.backend_latency_ms,
            postprocess_latency_ms=postprocess_latency_ms,
            raw_action=frame.values["_raw_action"],
            metadata={
                **backend_result.metadata,
                "pipeline_id": self._facade.pipeline_id,
                "bundle": manifest.bundle.name,
                "bundle_uuid": manifest.bundle.uuid,
                "bundle_revision": manifest.bundle.revision,
                "deployment": self._facade._context.deployment_name,
                "deployment_uuid": deployment.uuid,
                "deployment_revision": deployment.revision,
                "deployment_fingerprint": self._facade._context.deployment_fingerprint,
                "backend": self._facade._backend_name(),
                "state": self._facade.state.value,
                "latency_ms": latency_metadata,
            },
        )

    def adapt_error(self, error: ExecutionError) -> PipelineResult:
        if error.cause is not None:
            raise error.cause
        raise RuntimeError(error.message)


_CONTROL_INPUTS_KEY = "__ibrobot_policy_control_inputs"
_CAPTURE_RAW_ACTION_KEY = "__ibrobot_policy_capture_raw_action"
_PROMPT_KEY = "__ibrobot_policy_prompt"
_PRIORITY_KEY = "__ibrobot_policy_priority"


def _policy_contract(
    context: RuntimeContext, executor: SequentialModelExecutor | StagedModelExecutor
) -> ExecutionContract:
    """Derive the request contract for the migrated local policy executor."""

    model_type = context.model_type
    declared_contract = getattr(executor, "execution_contract", None)
    iterative = (
        declared_contract == "request-iterative"
        or model_type in {"diffusion", "pi05", "smolvla"}
        or any(hasattr(stage, "state_adapter") and hasattr(stage, "plan") for stage in executor.stages)
    )
    visibility = getattr(executor, "orchestration_visibility", None) or (
        "session" if model_type == "diffusion" else "executor"
    )
    return ExecutionContract(
        state_scope="request",
        execution_structure="iterative" if iterative else "direct",
        orchestration_visibility=visibility if iterative else None,
        cancellation_granularity="checkpoint" if iterative else "request_boundary",
    )


def _finalize_policy_assembly(
    assembly: RuntimeAssembly,
    context: RuntimeContext,
    executor: SequentialModelExecutor | StagedModelExecutor,
    *,
    resettable: bool,
) -> RuntimeAssembly:
    """Attach policy facade stages and ownership to a registered assembly."""

    profile = context.backend_profile
    if profile is None:
        raise PipelineConfigurationError(
            f"pipeline {context.deployment_name!r} has no typed runtime profile",
            pipeline_id=context.deployment_name,
            code="runtime_profile_required",
        )
    contract = _policy_contract(context, executor)
    capability_source = assembly.session or next(
        (
            component
            for component in executor.components
            if callable(getattr(component, "execute", None)) and callable(getattr(component, "health", None))
        ),
        None,
    )
    if capability_source is None:
        raise PipelineConfigurationError(
            f"pipeline {context.deployment_name!r} has no native session component",
            pipeline_id=context.deployment_name,
            code="runtime_session_required",
        )
    assembly.runtime_executor = executor
    assembly.executor = executor
    assembly.session = capability_source
    assembly.execution_contract = contract
    assembly.stateful = False
    # Independent execution owns snapshots even when the vendor session is stateless.
    assembly.resettable = resettable or isinstance(executor, StagedModelExecutor)
    assembly.declared_capabilities = {
        **dict(assembly.declared_capabilities),
        "stateful": False,
        "execution_contract": contract.name,
    }
    assembly.owned_components = tuple(
        OwnedComponent(
            component,
            f"policy_component:{index}",
            load_context=executor.component_contexts.get(id(component), context),
        )
        for index, component in enumerate(executor.components)
    ) + (OwnedComponent(executor, "policy_executor"),)
    return assembly


class _PolicySessionHandle:
    """Configuration bundle the factory passes so the facade can build the executor."""

    def __init__(
        self,
        model_executor,
        session_context: RuntimeContext,
        action_semantic: str,
        schedule_metadata: Mapping[str, object] | None,
        schedule: PI05DenoisingSchedule | None,
        capabilities,
        curvature_log_path: str | None,
        velocity_trace: list | None,
        *,
        preprocessor=None,
        postprocessor=None,
        metadata_provider=None,
    ) -> None:
        self.model_executor = model_executor
        self.session_context = session_context
        self.action_semantic = action_semantic
        self.schedule_metadata = schedule_metadata
        self.schedule = schedule
        self._capability_source = capabilities
        self.curvature_log_path = curvature_log_path
        self.velocity_trace = velocity_trace
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.metadata_provider = metadata_provider

    @property
    def capabilities(self):
        return getattr(self._capability_source, "capabilities", self._capability_source)


def _curvature_scores(velocities: list, eps: float = 1e-6) -> list[float]:
    if not velocities:
        return []
    if len(velocities) == 1:
        return [0.0]
    scores = []
    for current, following in pairwise(velocities):
        current_flat = np.asarray(current).reshape(np.asarray(current).shape[0], -1).astype(np.float32, copy=False)
        following_flat = (
            np.asarray(following).reshape(np.asarray(following).shape[0], -1).astype(np.float32, copy=False)
        )
        difference = np.linalg.norm(following_flat - current_flat, axis=1)
        magnitude = np.linalg.norm(current_flat, axis=1) + eps
        scores.append(float(np.mean(difference / magnitude)))
    scores.append(scores[-1])
    return scores


class InferencePipeline:
    """Policy facade over a unified handle or the legacy compatibility core.

    Preserves ``InferenceRequest``/``PipelineResult``, ``cancel()``, structured
    error mapping, prompt/control inputs, raw-action capture, codec/action
    validation, and stateful fail-closed semantics without owning any
    independent lifecycle, admission, or deadline state.
    """

    def __init__(
        self,
        pipeline_id: str,
        runtime_context: RuntimeContext,
        *,
        runtime_assembly: RuntimeAssembly,
        session_handle: _PolicySessionHandle,
        preprocessor: Processor | None = None,
        postprocessor: Postprocessor | None = None,
        codec: PolicyCodec | None = None,
        request_timeout: float | None = None,
        default_task: str | None = None,
        execution_mode: str = "monolithic",
        instrument_processors: bool = True,
        model_component_id: str = "policy.inference",
        stage_policy: str = "sequential",
        stage_scheduling: Mapping[str, object] | None = None,
        frame_base_priority: int = 0,
    ) -> None:
        if not pipeline_id:
            raise PipelineConfigurationError("pipeline_id must be non-empty")
        if execution_mode != "monolithic":
            raise PipelineConfigurationError(
                f"pipeline {pipeline_id!r} execution mode {execution_mode!r} is not implemented",
                pipeline_id=pipeline_id,
                code="unsupported_execution_mode",
                details={"execution_mode": execution_mode},
            )
        if request_timeout is not None and (not math.isfinite(request_timeout) or request_timeout <= 0):
            raise PipelineConfigurationError(
                f"pipeline {pipeline_id!r} request_timeout must be finite and positive",
                pipeline_id=pipeline_id,
                details={"request_timeout": request_timeout},
            )
        if not isinstance(runtime_assembly, RuntimeAssembly) or not isinstance(session_handle, _PolicySessionHandle):
            raise PipelineConfigurationError(
                f"pipeline {pipeline_id!r} requires a native policy runtime assembly",
                pipeline_id=pipeline_id,
                code="invalid_pipeline_construction",
            )

        self._pipeline_id = pipeline_id
        self._context = runtime_context
        self._runtime_assembly = runtime_assembly
        self._session_handle = session_handle
        self._pi05_handle = (
            session_handle
            if runtime_context.policy.policy_type == "pi05"
            and isinstance(runtime_context.deployment, CompiledDeployment)
            else None
        )
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._owns_preprocessor = preprocessor is not None
        self._owns_postprocessor = postprocessor is not None
        self._codec = codec
        self._request_timeout = request_timeout
        self._default_task = default_task
        self._execution_mode = execution_mode
        self._instrument_processors = instrument_processors is True
        self._model_component_id = model_component_id
        self._stage_policy = stage_policy
        self._stage_scheduling = stage_scheduling or {}
        self._frame_base_priority = frame_base_priority
        self._execution_plan: ExecutionPlan | None = None
        self._action_output_role: str | None = None
        self._policy_failure: BackendHealth | None = None
        self._unified_handle = None
        self._prepare_execution()
        self._bind_processors()

        resolved_executor: ModelExecutor = self._build_session_executor(session_handle)
        self._session_executor = resolved_executor
        _finalize_policy_assembly(
            runtime_assembly,
            runtime_context,
            resolved_executor,
            resettable=bool(self.capabilities.resettable),
        )
        runtime_assembly.reset_complete = self._clear_policy_failure if runtime_context.priority_scheduling else None
        runtime_assembly.retry_failed_reset = runtime_context.priority_scheduling
        self._unified_handle = ModelRuntimeHandle(runtime_assembly)
        if isinstance(resolved_executor, StagedModelExecutor):
            resolved_executor.frame_submitter = self._unified_handle.submit_frame
        self._pipeline = None

    def _build_session_executor(self, handle: _PolicySessionHandle) -> ModelExecutor:
        model_executor = handle.model_executor
        if not isinstance(model_executor, SequentialModelExecutor):
            raise PipelineConfigurationError(
                f"pipeline {self.pipeline_id!r} session factory must return SequentialModelExecutor",
                pipeline_id=self.pipeline_id,
                code="invalid_session_executor",
            )
        components = list(model_executor.components)
        if self._owns_preprocessor and self._preprocessor is not None:
            components.append(self._preprocessor)
        if self._owns_postprocessor and self._postprocessor is not None:
            components.append(self._postprocessor)
        component_contexts = dict(model_executor.component_contexts)
        component_contexts.update({id(component): handle.session_context for component in model_executor.components})
        stages = (
            _PolicyPreprocessStage(self),
            _PolicyModelRequestStage(self),
            *model_executor.stages,
            _PolicyModelResultStage(
                self,
                handle.action_semantic,
                handle.schedule_metadata,
                handle.metadata_provider,
            ),
            _PolicyDecodeStage(self),
            _PolicyPostprocessStage(self),
            _PolicyCompletionStage(lambda: self._write_curvature_log(handle)),
        )
        if self._stage_policy == "independent":
            deployment = self._context.deployment
            validate_staged = getattr(self._codec, "validate_staged_deployment", None)
            if not isinstance(deployment, CompiledDeployment) or not callable(validate_staged):
                raise PipelineConfigurationError(
                    "independent staged execution requires a validated policy-family adapter",
                    pipeline_id=self.pipeline_id,
                    code="staged_policy_adapter_required",
                )
            validate_staged(deployment)
            priorities = {
                stage_id: int(self._stage_option(value, "priority_offset", 0))
                for stage_id, value in self._stage_scheduling.items()
            }
            producer_role, terminal_role = deployment.execution
            return StagedModelExecutor(
                stages,
                _PolicyResultAdapter(self),
                components=components,
                execution_plan=build_execution_plan(deployment.execution, deployment.bindings),
                scheduling=StagedScheduling(
                    priorities,
                    {producer_role: "frame_arrival", terminal_role: "dispatch"},
                    frame_base_priority=self._frame_base_priority,
                    max_snapshot_age_ms=self._stage_option(
                        self._stage_scheduling.get(producer_role, {}), "max_snapshot_age_ms", 5000
                    ),
                ),
                component_contexts=component_contexts,
                error_handler=self._record_policy_failure,
                health_override=lambda: self._policy_failure,
            )
        return SequentialModelExecutor(
            stages,
            _PolicyResultAdapter(self),
            components=components,
            execution_plan=model_executor.execution_plan,
            component_contexts=component_contexts,
            error_handler=self._record_policy_failure,
            health_override=lambda: self._policy_failure,
            execution_contract=model_executor.execution_contract,
            orchestration_visibility=model_executor.orchestration_visibility,
        )

    @staticmethod
    def _stage_option(value: object, name: str, default: object) -> object:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    def _write_curvature_log(self, handle: _PolicySessionHandle) -> None:
        if handle.curvature_log_path is None or not handle.velocity_trace:
            return
        velocities = tuple(handle.velocity_trace)
        handle.velocity_trace.clear()
        record: dict[str, object] = {"curvature_scores": _curvature_scores(velocities)}
        if handle.schedule is not None:
            record["schedule"] = handle.schedule.to_dict()
        path = Path(str(handle.curvature_log_path)).expanduser().resolve()
        try:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
        except (OSError, ValueError) as exc:
            raise PipelineValidationError(
                f"pipeline {self.pipeline_id!r} unable to write PI0.5 curvature log {path}: {exc}",
                pipeline_id=self.pipeline_id,
                code="curvature_log_failed",
            ) from exc

    @property
    def pipeline_id(self) -> str:
        return self._pipeline_id

    @property
    def state(self):
        return self._pipeline_state_from_unified(self._unified_handle.state)

    @property
    def runtime_context(self) -> RuntimeContext:
        return self._context

    @property
    def runtime_handle(self) -> ModelRuntimeHandle | None:
        """Return the unified handle when this pipeline uses the migrated path."""

        return self._unified_handle

    @property
    def supports_priority_zero_deadline_admission(self) -> bool:
        return bool(self._session_executor.supports_priority_zero_deadline_admission)

    @property
    def capabilities(self):
        return self._session_handle.capabilities

    def load(self) -> None:
        try:
            self._unified_handle.load(self._context)
        except ExecutionFailure as exc:
            self._raise_execution_failure(exc)

    def infer(
        self,
        request: InferenceRequest,
        *,
        control_inputs: Mapping[str, object] | None = None,
        capture_raw_action: bool = False,
    ) -> PipelineResult:
        inputs = dict(request.inputs)
        inputs[_CONTROL_INPUTS_KEY] = dict(control_inputs or {})
        inputs[_CAPTURE_RAW_ACTION_KEY] = capture_raw_action
        inputs[_PROMPT_KEY] = request.prompt
        inputs[_PRIORITY_KEY] = request.priority
        unified_request = ModelRequest(inputs, request.metadata)
        context = ExecutionContext(request.request_id, self._effective_request_deadline(request.deadline))
        with (
            trace.trace_context(request.request_id, component_id=self._model_component_id)
            if trace.enabled
            else nullcontext()
        ):
            span_context = _model_span.set(None) if trace.enabled else None
            try:
                unified_result = self._unified_handle.execute(unified_request, context)
            except ExecutionFailure as exc:
                self._raise_execution_failure(exc)
            finally:
                if span_context is not None:
                    try:
                        pending_span = _model_span.get()
                        if pending_span is not None:
                            trace.end_span(pending_span, status="incomplete")
                    finally:
                        _model_span.reset(span_context)
        result = unified_result.outputs
        if isinstance(result, PipelineResult):
            result = replace(
                result,
                metadata={
                    **result.metadata,
                    "outcome_evidence": unified_result.evidence.to_dict(),
                },
            )
        if not isinstance(result, PipelineResult):
            raise PipelineValidationError(
                f"pipeline {self.pipeline_id!r} executor returned {type(result).__name__}, expected PipelineResult",
                pipeline_id=self.pipeline_id,
            )
        return result

    def submit_frame(self, request: InferenceRequest, *, deadline: datetime | None = None):
        submit = getattr(self._session_executor, "submit_frame", None)
        if not callable(submit):
            return None
        inputs = dict(request.inputs)
        inputs[_CONTROL_INPUTS_KEY] = {}
        inputs[_CAPTURE_RAW_ACTION_KEY] = False
        inputs[_PROMPT_KEY] = request.prompt
        inputs[_PRIORITY_KEY] = request.priority
        model_request = ModelRequest(
            inputs,
            {**request.metadata, "request_id": request.request_id},
        )
        return self._unified_handle.submit_frame(
            model_request,
            ExecutionContext(request.request_id, self._effective_request_deadline(deadline or request.deadline)),
        )

    def reset(self, deadline: datetime | None = None) -> None:
        if self.capabilities.stateful and not self.capabilities.resettable:
            raise PipelineLifecycleError(
                f"pipeline {self.pipeline_id!r} backend is stateful but does not support reset",
                pipeline_id=self.pipeline_id,
                code="reset_unsupported",
            )
        try:
            self._unified_handle.reset(deadline=Deadline.at(deadline))
        except ExecutionFailure as exc:
            self._raise_execution_failure(exc)
        health = self._unified_handle.health
        if not health.ready:
            raise PipelineNotReadyError(
                f"pipeline {self.pipeline_id!r} backend is not ready after reset",
                pipeline_id=self.pipeline_id,
                state=health.state.value,
            )

    def cancel(self, request_id: str, deadline: datetime | None = None) -> None:
        try:
            self._unified_handle.cancel(request_id, deadline=Deadline.at(deadline))
        except ExecutionFailure as exc:
            self._raise_execution_failure(exc)

    def diagnostics(self) -> PipelineDiagnostics:
        runtime_diag = self._unified_handle.diagnostics()
        identity = self._diagnostic_identity()
        health = self._unified_handle.health
        backend_health = BackendHealth(
            state=BackendState.READY if health.ready else BackendState.FAILED,
            ready=health.ready,
            reason_code=health.reason_code,
            message=health.message,
            recoverable=health.recoverable,
            failure_count=health.failure_count,
        )
        return PipelineDiagnostics(
            pipeline_id=self.pipeline_id,
            bundle=identity.bundle,
            bundle_uuid=identity.bundle_uuid,
            bundle_revision=identity.bundle_revision,
            deployment=identity.deployment,
            deployment_uuid=identity.deployment_uuid,
            deployment_revision=identity.deployment_revision,
            deployment_fingerprint=identity.deployment_fingerprint,
            backend=identity.backend,
            state=self.state,
            backend_health=backend_health,
            active_requests=runtime_diag.active_executions,
            request_timeout=self._request_timeout,
            default_task_configured=self._default_task is not None,
        )

    def health(self) -> PipelineDiagnostics:
        return self.diagnostics()

    def close(self) -> None:
        self._unified_handle.close()

    @staticmethod
    def _pipeline_state_from_unified(state: LifecycleState) -> object:
        if state is LifecycleState.RESET_REQUIRED:
            return PipelineState.DEGRADED
        try:
            return PipelineState(state.value)
        except ValueError:
            return PipelineState.FAILED

    def _diagnostic_identity(self):
        manifest = self._context.validated_manifest.manifest
        deployment = self._context.deployment
        return SimpleNamespace(
            bundle=manifest.bundle.name,
            bundle_uuid=manifest.bundle.uuid,
            bundle_revision=manifest.bundle.revision,
            deployment=self._context.deployment_name,
            deployment_uuid=deployment.uuid,
            deployment_revision=deployment.revision,
            deployment_fingerprint=self._context.deployment_fingerprint,
            backend=self._context.backend,
        )

    def _effective_request_deadline(self, requested: datetime | None) -> Deadline:
        if requested is not None:
            if requested.tzinfo is None:
                raise PipelineConfigurationError(
                    f"pipeline {self.pipeline_id!r} request deadline must be timezone-aware",
                    pipeline_id=self.pipeline_id,
                    code="invalid_deadline",
                )
            explicit = Deadline.at(requested)
        else:
            explicit = Deadline.unbounded()
        if self._request_timeout is None:
            return explicit
        configured = Deadline.after(self._request_timeout)
        if explicit.expires_at is None:
            return configured
        return Deadline.at(min(explicit.expires_at, configured.expires_at))

    @staticmethod
    def _raise_execution_failure(failure: ExecutionFailure) -> None:
        cause = failure.cause
        if isinstance(cause, Exception):
            raise cause from failure
        details = {
            **dict(failure.details),
            "evidence": failure.evidence.to_dict(),
            "recovery": failure.recovery.to_dict(),
        }
        raise PipelineLifecycleError(
            str(failure),
            pipeline_id="policy",
            code=failure.code,
            details=details,
        ) from failure

    # ------------------------------------------------------------------
    # Policy helpers (preserved from the original implementation)
    # ------------------------------------------------------------------

    @property
    def _action_dimension(self) -> int:
        return self._context.policy.output_features["action"].shape[-1]

    def _backend_name(self) -> str:
        return self._context.backend

    def _prepare_execution(self) -> None:
        deployment = self._context.deployment
        if not isinstance(deployment, CompiledDeployment) or not deployment.execution:
            self._execution_plan = None
            self._action_output_role = None
            return
        if self._codec is None:
            raise PipelineConfigurationError(
                f"compiled pipeline {self.pipeline_id!r} requires a policy codec",
                pipeline_id=self.pipeline_id,
                code="codec_required",
            )
        self._execution_plan = build_execution_plan(
            deployment.execution,
            deployment.bindings,
            deployment.device_links,
        )
        action_roles = [
            role
            for role in deployment.execution
            if any(binding.semantic == "action" for binding in deployment.bindings[role].outputs)
        ]
        if len(action_roles) != 1:
            raise PipelineConfigurationError(
                f"compiled pipeline {self.pipeline_id!r} must declare exactly one action-producing role",
                pipeline_id=self.pipeline_id,
                code="invalid_action_role",
                details={"action_roles": tuple(action_roles)},
            )
        self._action_output_role = action_roles[0]

    def _encode_role_inputs(self, canonical_inputs: Mapping[str, object]) -> Mapping[str, object]:
        codec, execution_plan = self._require_compiled_codec()
        encode_execution = getattr(codec, "encode_execution", None)
        if callable(encode_execution):
            return encode_execution(CodecRequest(canonical_inputs), execution_plan)
        if len(execution_plan.roles) == 1:
            role = execution_plan.roles[0]
            return {role.name: codec.encode_inputs(CodecRequest(canonical_inputs), role.bindings)}
        raise PipelineConfigurationError(
            f"compiled pipeline {self.pipeline_id!r} requires an execution-aware codec for multiple roles",
            pipeline_id=self.pipeline_id,
            code="execution_codec_required",
            details={"roles": execution_plan.role_names},
        )

    def _ensure_session_ready_after_call(self) -> None:
        if self._session_handle is None:
            return
        executor = self._session_executor if self._context.priority_scheduling else self._session_handle.model_executor
        health = executor.health()
        if not health.ready:
            raise PipelineNotReadyError(
                f"pipeline {self.pipeline_id!r} session left READY during inference",
                pipeline_id=self.pipeline_id,
                state=health.state.value,
            )

    def _decode_backend_action(self, result: _PolicyFrameResult) -> object:
        deployment = self._context.deployment
        if self._execution_plan is None:
            return result.action

        codec, execution_plan = self._require_compiled_codec()
        decode_execution = getattr(codec, "decode_execution", None)
        if callable(decode_execution):
            decoded = decode_execution(result.action, execution_plan)
        else:
            if self._action_output_role is None:
                raise PipelineConfigurationError(
                    f"compiled pipeline {self.pipeline_id!r} has no action output role",
                    pipeline_id=self.pipeline_id,
                    code="invalid_action_role",
                )
            decoded = codec.decode_outputs(result.action, deployment.bindings[self._action_output_role])
        if not isinstance(decoded, CodecResult):
            raise PipelineValidationError(
                f"pipeline {self.pipeline_id!r} codec returned {type(decoded).__name__}, expected CodecResult",
                pipeline_id=self.pipeline_id,
            )
        return decoded.action

    def _require_compiled_codec(self) -> tuple[object, ExecutionPlan]:
        if self._codec is None or self._execution_plan is None:
            raise PipelineConfigurationError(
                f"compiled pipeline {self.pipeline_id!r} is missing its codec or execution plan",
                pipeline_id=self.pipeline_id,
                code="codec_required",
            )
        return self._codec, self._execution_plan

    def _load_component(self, component: object) -> None:
        load = getattr(component, "load", None)
        if callable(load):
            load(self._context)

    def _bind_processors(self) -> None:
        if self._preprocessor is None:
            self._preprocessor = _identity_preprocessor
        if self._postprocessor is None:
            self._postprocessor = _identity_postprocessor

    def _raise_if_expired(self, deadline: datetime | None, *, phase: str, backend_completed: bool) -> None:
        if deadline is not None and datetime.now(timezone.utc) >= deadline:
            raise self._timeout_error(phase, backend_completed=backend_completed)

    def _timeout_error(self, phase: str, *, backend_completed: bool) -> PipelineTimeoutError:
        completion = " after the backend completed; the late result was discarded" if backend_completed else ""
        return PipelineTimeoutError(
            f"pipeline {self.pipeline_id!r} request deadline expired during {phase}{completion}",
            pipeline_id=self.pipeline_id,
            phase=phase,
            backend_completed=backend_completed,
            cancellation_supported=self.capabilities.supports_cancellation,
        )

    def _clear_policy_failure(self) -> None:
        self._policy_failure = None

    def _record_policy_failure(self, exc: Exception, backend_execution_started: bool) -> None:
        if not hasattr(exc, "operation_started"):
            exc.operation_started = bool(backend_execution_started)
        if not hasattr(exc, "outcome_known"):
            exc.outcome_known = True
        capabilities = self.capabilities
        if isinstance(exc, PipelineCanceledError):
            return
        if isinstance(exc, PipelineTimeoutError):
            if exc.details.get("backend_completed") and capabilities.stateful:
                self._policy_failure = BackendHealth(
                    state=BackendState.FAILED,
                    ready=False,
                    reason_code="deadline_exceeded",
                    message=str(exc),
                )
            return
        non_mutating_rejection = isinstance(exc, BackendAdmissionError) and not exc.operation_started
        if backend_execution_started and not non_mutating_rejection and capabilities.stateful:
            self._policy_failure = BackendHealth(
                state=BackendState.FAILED,
                ready=False,
                reason_code=getattr(exc, "code", "execution_failed"),
                message=str(exc),
            )

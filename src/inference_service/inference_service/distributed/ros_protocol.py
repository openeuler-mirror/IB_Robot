"""ROS message adapters for the transport-neutral distributed protocol."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from ibrobot_msgs.msg import (
    DistributedInferenceRequest,
    DistributedInferenceResult,
    InferenceError,
    InferenceFeatureSummary,
    InferencePipelineStatus,
    InferencePolicySummary,
    VariantsList,
)
from ibrobot_msgs.msg import VideoStreamDescriptor as VideoStreamDescriptorMessage
from ibrobot_msgs.msg import VideoStreamStatus as VideoStreamStatusMessage
from tensormsg.converter import TensorMsgConverter

from .types import (
    PROTOCOL_VERSION,
    DistributedRequest,
    DistributedResult,
    FeatureSummary,
    Operation,
    PeerRole,
    PipelineIdentity,
    PipelineStatus,
    PolicySummary,
    StreamReference,
    StructuredError,
)
from .video_streams import (
    VIDEO_DESCRIPTOR_FIELDS,
    VIDEO_STATUS_FIELDS,
    VideoStreamDescriptor,
    VideoStreamRuntimeStatus,
)


def status_to_message(status: PipelineStatus, *, stamp: Any | None = None) -> InferencePipelineStatus:
    message = InferencePipelineStatus()
    if stamp is not None:
        message.header.stamp = stamp
    message.protocol_version = status.identity.protocol_version
    message.role = int(status.role)
    message.pipeline_id = status.identity.pipeline_id
    message.manifest_schema_version = status.identity.manifest_schema_version
    message.bundle_uuid = status.identity.bundle_uuid
    message.bundle_revision = status.identity.bundle_revision
    message.bundle_digest = status.identity.bundle_digest
    message.deployment_name = status.identity.deployment_name
    message.deployment_uuid = status.identity.deployment_uuid
    message.deployment_revision = status.identity.deployment_revision
    message.deployment_fingerprint = status.identity.deployment_fingerprint
    message.policy_summary = _policy_to_message(status.identity.policy)
    message.sequence = status.sequence
    message.session_id = status.session_id
    message.session_generation = status.session_generation
    message.ready = status.ready
    message.runtime_state = status.runtime_state
    message.reset_supported = status.reset_supported
    message.cancellation_supported = status.cancellation_supported
    message.error = error_to_message(status.error)
    return message


def status_from_message(message: InferencePipelineStatus) -> PipelineStatus:
    return PipelineStatus(
        role=PeerRole(message.role),
        identity=PipelineIdentity(
            pipeline_id=message.pipeline_id,
            manifest_schema_version=message.manifest_schema_version,
            bundle_uuid=message.bundle_uuid,
            bundle_revision=message.bundle_revision,
            bundle_digest=message.bundle_digest,
            deployment_name=message.deployment_name,
            deployment_uuid=message.deployment_uuid,
            deployment_revision=message.deployment_revision,
            deployment_fingerprint=message.deployment_fingerprint,
            policy=_policy_from_message(message.policy_summary),
            protocol_version=message.protocol_version,
        ),
        sequence=message.sequence,
        session_id=message.session_id,
        session_generation=message.session_generation,
        ready=message.ready,
        runtime_state=message.runtime_state,
        reset_supported=message.reset_supported,
        cancellation_supported=message.cancellation_supported,
        error=error_from_message(message.error),
    )


def request_to_message(request: DistributedRequest) -> DistributedInferenceRequest:
    message = DistributedInferenceRequest()
    message.protocol_version = PROTOCOL_VERSION
    message.operation = int(request.operation)
    message.pipeline_id = request.pipeline_id
    message.request_id = request.request_id
    message.target_request_id = request.target_request_id
    message.session_id = request.session_id
    message.session_generation = request.session_generation
    message.deployment_fingerprint = request.deployment_fingerprint
    if request.deadline is not None:
        seconds = request.deadline.astimezone(timezone.utc).timestamp()
        message.deadline.sec = int(seconds)
        message.deadline.nanosec = int((seconds - int(seconds)) * 1_000_000_000)
    message.prompt = request.prompt or ""
    _nanoseconds_to_time(request.observation_timestamp_ns, message.observation_timestamp)
    message.stream_observation_keys = [reference.observation_key for reference in request.stream_references]
    message.stream_ids = [reference.stream_id for reference in request.stream_references]
    message.tensors = TensorMsgConverter.to_variant(dict(request.inputs))
    return message


def request_from_message(message: DistributedInferenceRequest) -> DistributedRequest:
    if message.protocol_version != PROTOCOL_VERSION:
        raise ValueError(
            f"unsupported distributed protocol version {message.protocol_version}; expected {PROTOCOL_VERSION}"
        )
    deadline = _datetime_from_message(message.deadline.sec, message.deadline.nanosec)
    if len(message.stream_observation_keys) != len(message.stream_ids):
        raise ValueError("distributed request stream reference arrays must have equal length")
    return DistributedRequest(
        operation=Operation(message.operation),
        pipeline_id=message.pipeline_id,
        request_id=message.request_id,
        target_request_id=message.target_request_id,
        session_id=message.session_id,
        session_generation=message.session_generation,
        deployment_fingerprint=message.deployment_fingerprint,
        inputs=TensorMsgConverter.from_variant(message.tensors),
        prompt=message.prompt or None,
        deadline=deadline,
        observation_timestamp_ns=_time_to_nanoseconds(message.observation_timestamp),
        stream_references=tuple(
            StreamReference(observation_key, stream_id)
            for observation_key, stream_id in zip(
                message.stream_observation_keys,
                message.stream_ids,
                strict=True,
            )
        ),
    )


def video_descriptor_to_message(
    descriptor: VideoStreamDescriptor, *, stamp: Any | None = None
) -> VideoStreamDescriptorMessage:
    message = VideoStreamDescriptorMessage()
    if stamp is not None:
        message.header.stamp = stamp
    for field_name in VIDEO_DESCRIPTOR_FIELDS:
        setattr(message, field_name, getattr(descriptor, field_name))
    return message


def video_descriptor_from_message(message: VideoStreamDescriptorMessage) -> VideoStreamDescriptor:
    return VideoStreamDescriptor(**{field_name: getattr(message, field_name) for field_name in VIDEO_DESCRIPTOR_FIELDS})


def video_status_to_message(status: VideoStreamRuntimeStatus, *, stamp: Any | None = None) -> VideoStreamStatusMessage:
    message = VideoStreamStatusMessage()
    if stamp is not None:
        message.header.stamp = stamp
    for field_name in VIDEO_STATUS_FIELDS:
        setattr(message, field_name, getattr(status, field_name))
    _nanoseconds_to_time(status.mapping_capture_timestamp_ns, message.mapping_capture_time)
    return message


def video_status_from_message(message: VideoStreamStatusMessage) -> VideoStreamRuntimeStatus:
    return VideoStreamRuntimeStatus(
        **{field_name: getattr(message, field_name) for field_name in VIDEO_STATUS_FIELDS},
        mapping_capture_timestamp_ns=_time_to_nanoseconds(message.mapping_capture_time),
    )


def result_to_message(result: DistributedResult) -> DistributedInferenceResult:
    message = DistributedInferenceResult()
    message.protocol_version = PROTOCOL_VERSION
    message.operation = int(result.operation)
    message.pipeline_id = result.pipeline_id
    message.request_id = result.request_id
    message.target_request_id = result.target_request_id
    message.session_id = result.session_id
    message.session_generation = result.session_generation
    message.deployment_fingerprint = result.deployment_fingerprint
    message.success = result.success
    message.error = error_to_message(result.error)
    message.action_chunk = (
        TensorMsgConverter.to_variant({"action": result.action}) if result.action is not None else VariantsList()
    )
    message.actual_chunk_size = result.actual_chunk_size
    message.execution_horizon = result.execution_horizon
    message.backend_latency_ms = result.backend_latency_ms
    message.performance_json = json.dumps(dict(result.performance), sort_keys=True, separators=(",", ":"))
    message.backend_ready = result.backend_ready
    message.backend_state = result.backend_state
    return message


def result_from_message(message: DistributedInferenceResult) -> DistributedResult:
    if message.protocol_version != PROTOCOL_VERSION:
        raise ValueError(
            f"unsupported distributed protocol version {message.protocol_version}; expected {PROTOCOL_VERSION}"
        )
    decoded = TensorMsgConverter.from_variant(message.action_chunk) if message.action_chunk.variants else {}
    performance = json.loads(message.performance_json or "{}")
    if not isinstance(performance, dict):
        raise ValueError("distributed result performance_json must contain an object")
    return DistributedResult(
        operation=Operation(message.operation),
        pipeline_id=message.pipeline_id,
        request_id=message.request_id,
        target_request_id=message.target_request_id,
        session_id=message.session_id,
        session_generation=message.session_generation,
        deployment_fingerprint=message.deployment_fingerprint,
        success=message.success,
        action=decoded.get("action"),
        actual_chunk_size=message.actual_chunk_size,
        execution_horizon=message.execution_horizon,
        backend_latency_ms=message.backend_latency_ms,
        performance=performance,
        backend_ready=message.backend_ready,
        backend_state=message.backend_state,
        error=error_from_message(message.error),
    )


def error_to_message(error: StructuredError | None) -> InferenceError:
    message = InferenceError()
    if error is None:
        return message
    message.code = error.code
    message.message = error.message
    message.recoverable = error.recoverable
    message.stage = error.stage
    message.details_json = json.dumps(dict(error.details), sort_keys=True, separators=(",", ":"), default=_json_default)
    return message


def error_from_message(message: InferenceError) -> StructuredError | None:
    if not message.code:
        return None
    details = json.loads(message.details_json) if message.details_json else {}
    if not isinstance(details, dict):
        raise ValueError("InferenceError.details_json must contain a JSON object")
    return StructuredError(
        code=message.code,
        message=message.message,
        recoverable=message.recoverable,
        stage=message.stage,
        details=details,
    )


def decode_failure_result(
    message: DistributedInferenceRequest,
    exc: Exception,
    fingerprint: str,
    pipeline_id: str,
) -> DistributedResult:
    return transport_failure_result(
        message,
        exc,
        fingerprint,
        pipeline_id,
        code="decode_failed",
        stage="decode",
    )


def transport_failure_result(
    message: DistributedInferenceRequest,
    exc: Exception,
    fingerprint: str,
    pipeline_id: str,
    *,
    code: str,
    stage: str,
    operation: Operation | None = None,
) -> DistributedResult:
    error = StructuredError(
        code=code,
        message=str(exc) or type(exc).__name__,
        stage=stage,
    )
    if operation is None:
        try:
            operation = Operation(message.operation)
        except ValueError:
            operation = Operation.UNKNOWN
    return DistributedResult(
        operation=operation,
        pipeline_id=pipeline_id,
        request_id=message.request_id or "unknown",
        target_request_id=message.target_request_id,
        session_id=message.session_id,
        session_generation=message.session_generation,
        deployment_fingerprint=fingerprint,
        success=False,
        backend_ready=False,
        backend_state=code,
        error=error,
    )


def _policy_to_message(summary: PolicySummary) -> InferencePolicySummary:
    message = InferencePolicySummary()
    message.policy_type = summary.policy_type
    message.inputs = [_feature_to_message(feature) for feature in summary.inputs]
    message.outputs = [_feature_to_message(feature) for feature in summary.outputs]
    message.action_dimension = summary.action_dimension
    message.nominal_chunk_size = summary.nominal_chunk_size or 0
    return message


def _policy_from_message(message: InferencePolicySummary) -> PolicySummary:
    return PolicySummary(
        policy_type=message.policy_type,
        inputs=tuple(_feature_from_message(feature) for feature in message.inputs),
        outputs=tuple(_feature_from_message(feature) for feature in message.outputs),
        action_dimension=message.action_dimension,
        nominal_chunk_size=message.nominal_chunk_size or None,
    )


def _feature_to_message(summary: FeatureSummary) -> InferenceFeatureSummary:
    message = InferenceFeatureSummary()
    message.semantic = summary.semantic
    message.feature_type = summary.feature_type
    message.shape = list(summary.shape)
    return message


def _feature_from_message(message: InferenceFeatureSummary) -> FeatureSummary:
    return FeatureSummary(
        semantic=message.semantic,
        feature_type=message.feature_type,
        shape=tuple(message.shape),
    )


def _datetime_from_message(seconds: int, nanoseconds: int) -> datetime | None:
    if seconds == 0 and nanoseconds == 0:
        return None
    return datetime.fromtimestamp(seconds + nanoseconds / 1_000_000_000, tz=timezone.utc)


def _nanoseconds_to_time(timestamp_ns: int, message: Any) -> None:
    message.sec, message.nanosec = divmod(timestamp_ns, 1_000_000_000)


def _time_to_nanoseconds(message: Any) -> int:
    return int(message.sec) * 1_000_000_000 + int(message.nanosec)


def _json_default(value: object) -> object:
    try:
        return asdict(value)  # type: ignore[arg-type]
    except TypeError:
        return str(value)

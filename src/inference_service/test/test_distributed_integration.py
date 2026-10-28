from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inference_manifest import load_inference_manifest
from inference_service.backends import BackendCapabilities, BackendHealth, BackendState
from inference_service.distributed import (
    DistributedCloudService,
    DistributedRequest,
    DistributedResult,
    EdgeSession,
    Operation,
    StructuredError,
    build_pipeline_identity,
)
from inference_service.distributed.ros_protocol import (
    request_from_message,
    request_to_message,
    result_from_message,
    result_to_message,
    status_from_message,
    status_to_message,
)
from inference_service.distributed.runtime import EdgeProcessorRuntime
from inference_service.pipeline import PipelineState
from inference_service.pipeline_policy_node import PipelinePolicyNode, _RoundTripProgress
from test.manifest_fixtures import create_policy_bundle, make_manifest, write_manifest


def _identity(root: Path, pipeline_id: str = "policy"):
    root.mkdir()
    paths = create_policy_bundle(root)
    config_path = root / "config.json"
    config = config_path.read_text(encoding="utf-8").replace(
        '"device": "cuda"',
        '"chunk_size": 4,\n  "device": "cuda"',
    )
    config_path.write_text(config, encoding="utf-8")
    write_manifest(root, make_manifest(root, paths))
    return build_pipeline_identity(pipeline_id, load_inference_manifest(root, "cpu"))


class _BackendFailure(RuntimeError):
    code = "device_lost"
    recoverable = True


class _MockCloudRuntime:
    def __init__(self) -> None:
        self.capabilities = BackendCapabilities(
            resettable=True,
            stateful=True,
            supports_cancellation=True,
        )
        self.state = BackendState.READY
        self.infer_error: Exception | None = None
        self.infer_calls: list[str] = []
        self.reset_calls = 0
        self.canceled_request_ids: list[str] = []
        self.closed = False

    def health(self) -> BackendHealth:
        return BackendHealth(
            state=self.state,
            ready=self.state is BackendState.READY,
            reason_code=None if self.state is BackendState.READY else "device_lost",
            message=None if self.state is BackendState.READY else "mock accelerator unavailable",
            recoverable=self.state is not BackendState.FAILED,
        )

    def infer(self, request_id, _inputs, *, prompt=None, deadline=None):
        self.infer_calls.append(request_id)
        if self.infer_error is not None:
            raise self.infer_error
        return SimpleNamespace(
            action=np.arange(12, dtype=np.float32).reshape(2, 6),
            actual_chunk_size=2,
            backend_latency_ms=3.25,
        )

    def reset(self, deadline=None) -> None:
        self.reset_calls += 1

    def cancel(self, request_id: str, deadline=None) -> None:
        self.canceled_request_ids.append(request_id)

    def close(self) -> None:
        self.closed = True


class _RosTransportHarness:
    def __init__(self, identity, runtime: _MockCloudRuntime, *, heartbeat_timeout: float = 1.0) -> None:
        self.edge = EdgeSession(identity, heartbeat_timeout=heartbeat_timeout)
        self.cloud = DistributedCloudService(identity, runtime)
        self.now = 10.0

    def start(self):
        self.edge.start()
        return self.exchange_status()

    def exchange_status(self):
        edge_status = status_from_message(status_to_message(self.edge.local_status()))
        cloud_status = self.cloud.observe_edge(edge_status)
        return self.edge.observe_cloud(status_from_message(status_to_message(cloud_status)), now=self.now)

    def round_trip(
        self,
        operation: Operation,
        request_id: str,
        *,
        inputs: dict[str, object] | None = None,
        deadline: datetime | None = None,
        target_request_id: str = "",
        mutate_request=None,
    ):
        observed = {}

        def sender(request: DistributedRequest) -> None:
            decoded_request = request_from_message(request_to_message(request))
            if mutate_request is not None:
                decoded_request = mutate_request(decoded_request)
            cloud_result = self.cloud.handle(decoded_request)
            result = result_from_message(result_to_message(cloud_result))
            observed["result"] = result
            observed["update"] = self.edge.accept_result(result)

        request = self.edge.dispatch_request(
            operation,
            request_id,
            sender,
            inputs=inputs,
            deadline=deadline,
            target_request_id=target_request_id,
        )
        return request, observed["result"], observed["update"]


def test_ros_transport_matching_startup_and_inference_metadata(tmp_path):
    identity = _identity(tmp_path / "bundle")
    runtime = _MockCloudRuntime()
    transport = _RosTransportHarness(identity, runtime)

    update = transport.start()
    request, result, result_update = transport.round_trip(
        Operation.INFER,
        "infer-1",
        inputs={"observation.state": np.zeros((1, 6), dtype=np.float32)},
        deadline=datetime.now(timezone.utc) + timedelta(seconds=1),
    )

    assert update.error is None
    assert transport.edge.ready
    assert request.pipeline_id == identity.pipeline_id
    assert result.success
    assert result.actual_chunk_size == 2
    assert result.backend_latency_ms == 3.25
    assert result.deployment_fingerprint == identity.deployment_fingerprint
    assert np.array_equal(result.action, np.arange(12, dtype=np.float32).reshape(2, 6))
    assert result_update.error is None
    assert runtime.infer_calls == ["infer-1"]


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("bundle_uuid", "bundle_uuid_mismatch"),
        ("bundle_revision", "bundle_revision_mismatch"),
        ("bundle_digest", "bundle_digest_mismatch"),
        ("deployment_name", "deployment_mismatch"),
        ("deployment_uuid", "deployment_uuid_mismatch"),
        ("deployment_revision", "deployment_revision_mismatch"),
        ("deployment_fingerprint", "deployment_fingerprint_mismatch"),
    ],
)
def test_ros_transport_rejects_bundle_and_deployment_mismatches(tmp_path, field, code):
    edge_identity = _identity(tmp_path / "bundle")
    current = getattr(edge_identity, field)
    cloud_identity = replace(
        edge_identity,
        **{field: current + 1 if isinstance(current, int) else f"different-{field}"},
    )
    runtime = _MockCloudRuntime()
    edge = EdgeSession(edge_identity)
    cloud = DistributedCloudService(cloud_identity, runtime)
    edge.start()

    edge_status = status_from_message(status_to_message(edge.local_status()))
    cloud_status = status_from_message(status_to_message(cloud.observe_edge(edge_status)))
    update = edge.observe_cloud(cloud_status)

    assert update.error is not None
    assert update.error.code == code
    assert edge.state is PipelineState.HANDSHAKING
    assert runtime.infer_calls == []


def test_ros_transport_returns_unknown_pipeline_and_backend_errors_immediately(tmp_path):
    identity = _identity(tmp_path / "bundle")
    runtime = _MockCloudRuntime()
    transport = _RosTransportHarness(identity, runtime)
    transport.start()

    _, unknown, unknown_update = transport.round_trip(
        Operation.INFER,
        "unknown-pipeline",
        mutate_request=lambda request: replace(request, pipeline_id="missing"),
    )

    assert not unknown.success
    assert unknown.error is not None
    assert unknown.error.code == "pipeline_not_found"
    assert unknown_update.error is None
    assert runtime.infer_calls == []

    runtime.infer_error = RuntimeError("tensor validation failed")
    _, failed, failed_update = transport.round_trip(Operation.INFER, "backend-error")

    assert not failed.success
    assert failed.error is not None
    assert failed.error.code == "operation_failed"
    assert failed.error.stage == "backend"
    assert failed_update.error is None


def test_ros_transport_heartbeat_loss_requires_a_new_handshake(tmp_path):
    identity = _identity(tmp_path / "bundle")
    runtime = _MockCloudRuntime()
    transport = _RosTransportHarness(identity, runtime)
    transport.start()
    first_session = transport.edge.session
    transport.edge.prepare_request(Operation.INFER, "in-flight")

    expired = transport.edge.expire_heartbeat(now=transport.now + 1.1)

    assert expired.invalidated_request_ids == ("in-flight",)
    assert expired.error is not None
    assert expired.error.code == "heartbeat_expired"
    assert not transport.edge.ready

    transport.now += 1.2
    recovered = transport.exchange_status()

    assert recovered.error is None
    assert transport.edge.ready
    assert transport.edge.session != first_session
    assert transport.edge.session[1] > first_session[1]


def test_ros_transport_cloud_restart_discards_stale_response(tmp_path):
    identity = _identity(tmp_path / "bundle")
    runtime = _MockCloudRuntime()
    transport = _RosTransportHarness(identity, runtime)
    transport.start()
    old_request = transport.edge.prepare_request(Operation.INFER, "old")

    transport.cloud = DistributedCloudService(identity, runtime)
    restarted = transport.exchange_status()

    assert restarted.invalidated_request_ids == ("old",)
    assert transport.edge.ready
    current = transport.edge.prepare_request(Operation.INFER, "current")
    stale_result = DistributedResult(
        operation=Operation.INFER,
        pipeline_id=identity.pipeline_id,
        request_id=old_request.request_id,
        session_id=old_request.session_id,
        session_generation=old_request.session_generation,
        deployment_fingerprint=identity.deployment_fingerprint,
        success=True,
        action=np.zeros((1, 6), dtype=np.float32),
        actual_chunk_size=1,
        backend_ready=True,
        backend_state="ready",
    )

    stale = result_from_message(result_to_message(stale_result))
    stale_update = transport.edge.accept_result(stale)

    assert stale_update.error is not None
    assert stale_update.error.code == "stale_response"
    assert current.request_id == "current"


def test_ros_transport_backend_failure_revokes_and_rehandshakes(tmp_path):
    identity = _identity(tmp_path / "bundle")
    runtime = _MockCloudRuntime()
    transport = _RosTransportHarness(identity, runtime)
    transport.start()
    first_session = transport.edge.session
    runtime.state = BackendState.FAILED
    runtime.infer_error = _BackendFailure("accelerator disappeared")

    _, result, update = transport.round_trip(Operation.INFER, "failure")

    assert not result.success
    assert result.error is not None
    assert result.error.code == "device_lost"
    assert update.error is not None
    assert update.error.code == "remote_backend_unavailable"
    assert transport.edge.state is PipelineState.DEGRADED

    runtime.state = BackendState.READY
    runtime.infer_error = None
    recovered = transport.exchange_status()

    assert recovered.error is None
    assert transport.edge.ready
    assert transport.edge.session != first_session


def test_ros_transport_timeout_reset_and_cancellation(tmp_path):
    identity = _identity(tmp_path / "bundle")
    runtime = _MockCloudRuntime()
    transport = _RosTransportHarness(identity, runtime)
    transport.start()

    _, expired, expired_update = transport.round_trip(
        Operation.INFER,
        "expired",
        deadline=datetime.now(timezone.utc) + timedelta(seconds=1),
        mutate_request=lambda request: replace(
            request,
            deadline=datetime.now(timezone.utc) - timedelta(seconds=1),
        ),
    )

    assert not expired.success
    assert expired.error is not None
    assert expired.error.code == "deadline_exceeded"
    assert expired_update.error is None
    assert runtime.infer_calls == []

    _, reset, reset_update = transport.round_trip(Operation.RESET, "reset")

    assert reset.success
    assert reset_update.error is None
    assert runtime.reset_calls == 1

    transport.edge.prepare_request(Operation.INFER, "target")
    _, canceled, cancel_update = transport.round_trip(
        Operation.CANCEL,
        "cancel",
        target_request_id="target",
    )

    assert canceled.success
    assert cancel_update.error is None
    assert cancel_update.canceled_request_id == "target"
    assert runtime.canceled_request_ids == ["target"]


def test_ros_transport_cloud_startup_dependency_error_is_immediate(tmp_path):
    identity = _identity(tmp_path / "bundle")
    startup_error = StructuredError(
        code="missing_dependency",
        message="accelerator SDK is unavailable",
        stage="startup",
    )
    edge = EdgeSession(identity)
    cloud = DistributedCloudService(identity, None, startup_error=startup_error)
    edge.start()
    cloud_status = cloud.observe_edge(status_from_message(status_to_message(edge.local_status())))
    update = edge.observe_cloud(status_from_message(status_to_message(cloud_status)))

    assert update.error == startup_error
    request = DistributedRequest(
        operation=Operation.RESET,
        pipeline_id=identity.pipeline_id,
        request_id="startup-error",
        session_id="unavailable",
        session_generation=1,
        deployment_fingerprint=identity.deployment_fingerprint,
    )
    result = result_from_message(result_to_message(cloud.handle(request_from_message(request_to_message(request)))))

    assert not result.success
    assert result.error == startup_error


def test_edge_session_failure_is_terminal_and_invalidates_pending_requests(tmp_path):
    identity = _identity(tmp_path / "bundle")
    runtime = _MockCloudRuntime()
    transport = _RosTransportHarness(identity, runtime)
    transport.start()
    transport.edge.prepare_request(Operation.INFER, "pending")
    failure = StructuredError(code="processor_reset_failed", message="edge reset failed", stage="reset")

    update = transport.edge.fail(failure)
    observed = transport.exchange_status()

    assert update.invalidated_request_ids == ("pending",)
    assert update.error == failure
    assert observed.error == failure
    assert transport.edge.state is PipelineState.FAILED
    assert not transport.edge.ready


def test_edge_processor_runtime_has_no_lerobot_processor_state():
    runtime = EdgeProcessorRuntime.__new__(EdgeProcessorRuntime)
    runtime.pipeline_id = "policy"
    runtime._loaded = True
    runtime._default_task = None
    runtime._manifest = SimpleNamespace(policy=SimpleNamespace(output_features={"action": SimpleNamespace(shape=(6,))}))

    inputs = {"observation.state": np.zeros((1, 6), dtype=np.float32)}
    assert runtime.preprocess(inputs) == inputs
    runtime.reset()
    assert runtime.postprocess(np.zeros((1, 6), dtype=np.float32), actual_chunk_size=1).shape == (1, 6)


def test_edge_processor_runtime_deadline_after_successful_reset_does_not_poison_runtime():
    runtime = EdgeProcessorRuntime.__new__(EdgeProcessorRuntime)
    runtime._loaded = True
    runtime._reset_error = None
    runtime._default_task = None
    runtime._preprocessor = SimpleNamespace(
        reset=lambda: None,
        __call__=lambda values: values,
    )

    with pytest.raises(TimeoutError, match="deadline expired"):
        runtime.reset(datetime.now(timezone.utc) - timedelta(seconds=1))

    assert runtime._reset_error is None


def test_distributed_reset_failure_before_edge_reset_is_not_terminal():
    edge_runtime = SimpleNamespace(reset_calls=0)
    edge_runtime.reset = lambda: setattr(edge_runtime, "reset_calls", edge_runtime.reset_calls + 1)
    node = SimpleNamespace(
        _config=SimpleNamespace(pipeline_id="policy", request_timeout=0.1),
        _require_edge_session=lambda: SimpleNamespace(reset_supported=True),
        _round_trip=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("cloud reset failed")),
        _require_edge_runtime=lambda: edge_runtime,
    )

    error, terminal = PipelinePolicyNode._reset_distributed_pipeline(node)

    assert str(error) == "cloud reset failed"
    assert terminal is False
    assert edge_runtime.reset_calls == 0


def test_distributed_reset_unknown_outcome_is_terminal_after_request_publication():
    edge_runtime = SimpleNamespace(reset_calls=0)

    def fail_round_trip(*_args, **kwargs):
        kwargs["progress"].published = True
        raise RuntimeError("reset result timed out")

    node = SimpleNamespace(
        _config=SimpleNamespace(pipeline_id="policy", request_timeout=0.1),
        _require_edge_session=lambda: SimpleNamespace(reset_supported=True),
        _round_trip=fail_round_trip,
        _require_edge_runtime=lambda: edge_runtime,
    )

    error, terminal = PipelinePolicyNode._reset_distributed_pipeline(node)

    assert str(error) == "reset result timed out"
    assert terminal is True


def test_distributed_reset_definitive_ready_failure_is_recoverable():
    def fail_round_trip(*_args, **kwargs):
        progress: _RoundTripProgress = kwargs["progress"]
        progress.published = True
        progress.response_received = True
        progress.backend_ready = True
        raise RuntimeError("reset rejected")

    node = SimpleNamespace(
        _config=SimpleNamespace(pipeline_id="policy", request_timeout=0.1),
        _require_edge_session=lambda: SimpleNamespace(reset_supported=True),
        _round_trip=fail_round_trip,
    )

    error, terminal = PipelinePolicyNode._reset_distributed_pipeline(node)

    assert str(error) == "reset rejected"
    assert terminal is False


def test_distributed_edge_reset_failure_after_cloud_success_is_terminal():
    edge_runtime = SimpleNamespace(reset=lambda _deadline: (_ for _ in ()).throw(RuntimeError("edge reset failed")))
    node = SimpleNamespace(
        _config=SimpleNamespace(pipeline_id="policy", request_timeout=0.1),
        _require_edge_session=lambda: SimpleNamespace(reset_supported=True),
        _round_trip=lambda *_args, **_kwargs: None,
        _require_edge_runtime=lambda: edge_runtime,
    )

    error, terminal = PipelinePolicyNode._reset_distributed_pipeline(node)

    assert str(error) == "edge reset failed"
    assert terminal is True

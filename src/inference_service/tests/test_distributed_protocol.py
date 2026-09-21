from __future__ import annotations

import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inference_manifest import load_inference_manifest
from inference_service.backends import BackendCancellationError
from inference_service.distributed import (
    CloudSession,
    DistributedCloudService,
    DistributedProtocolError,
    DistributedRequest,
    DistributedResult,
    EdgeSession,
    Operation,
    PeerRole,
    PipelineStatus,
    StreamReference,
    StructuredError,
    build_pipeline_identity,
)
from inference_service.distributed.ros_protocol import (
    decode_failure_result,
    request_from_message,
    request_to_message,
    result_from_message,
    result_to_message,
    status_from_message,
    status_to_message,
)
from inference_service.distributed.service import align_inputs_to_selection
from inference_service.distributed.types import PROTOCOL_VERSION, structured_error_from_exception
from inference_service.pipeline import PipelineState
from tests.manifest_fixtures import create_policy_bundle, make_manifest, write_manifest


def _identity(root: Path, pipeline_id: str = "policy"):
    root.mkdir()
    paths = create_policy_bundle(root)
    config_path = root / "config.json"
    config = config_path.read_text(encoding="utf-8").replace('"device": "cuda"', '"chunk_size": 4,\n  "device": "cuda"')
    config_path.write_text(config, encoding="utf-8")
    write_manifest(root, make_manifest(root, paths))
    return build_pipeline_identity(pipeline_id, load_inference_manifest(root, "cpu"))


def _handshake(edge: EdgeSession, cloud: CloudSession) -> PipelineStatus:
    edge.start()
    edge_status = edge.local_status()
    assert cloud.observe_edge(edge_status, backend_ready=True) is None
    cloud_status = cloud.status(
        backend_ready=True,
        backend_state="ready",
        reset_supported=True,
        cancellation_supported=True,
    )
    update = edge.observe_cloud(cloud_status, now=10.0)
    assert update.error is None
    assert edge.ready
    return cloud_status


def test_identity_summary_is_canonical_and_includes_chunk_contract(tmp_path):
    identity = _identity(tmp_path / "bundle")

    assert identity.pipeline_id == "policy"
    assert [feature.semantic for feature in identity.policy.inputs] == [
        "observation.images.top",
        "observation.state",
    ]
    assert identity.policy.action_dimension == 6
    assert identity.policy.nominal_chunk_size == 4
    assert identity.bundle_uuid
    assert identity.bundle_revision == 1
    assert identity.deployment_uuid
    assert identity.deployment_revision == 1


def test_matching_handshake_gates_requests_and_routes_result(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    cloud = CloudSession(identity)

    with pytest.raises(DistributedProtocolError) as not_ready:
        edge.prepare_request(Operation.INFER, "request-1")
    assert not_ready.value.code == "not_ready"

    cloud_status = _handshake(edge, cloud)
    request = edge.prepare_request(
        Operation.INFER,
        "request-1",
        inputs={"observation.state": np.zeros((1, 6), dtype=np.float32)},
        deadline=datetime.now(timezone.utc) + timedelta(seconds=1),
        observation_timestamp_ns=1_000_000_000,
        stream_references=(StreamReference("observation.images.top", "top"),),
    )
    cloud.validate_request(request)
    result = DistributedResult(
        operation=Operation.INFER,
        pipeline_id=identity.pipeline_id,
        request_id=request.request_id,
        session_id=cloud_status.session_id,
        session_generation=cloud_status.session_generation,
        deployment_fingerprint=identity.deployment_fingerprint,
        success=True,
        action=np.zeros((2, 6), dtype=np.float32),
        actual_chunk_size=2,
        execution_horizon=1,
        backend_latency_ms=1.5,
        backend_ready=True,
        backend_state="ready",
    )

    update = edge.accept_result(result)

    assert update.error is None


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("protocol_version", "protocol_version_mismatch"),
        ("pipeline_id", "pipeline_id_mismatch"),
        ("bundle_uuid", "bundle_uuid_mismatch"),
        ("bundle_revision", "bundle_revision_mismatch"),
        ("bundle_digest", "bundle_digest_mismatch"),
        ("deployment_name", "deployment_mismatch"),
        ("deployment_uuid", "deployment_uuid_mismatch"),
        ("deployment_revision", "deployment_revision_mismatch"),
        ("deployment_fingerprint", "deployment_fingerprint_mismatch"),
    ],
)
def test_handshake_rejects_identity_mismatches(tmp_path, field, code):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    edge.start()
    current = getattr(identity, field)
    remote_identity = replace(identity, **{field: current + 1 if isinstance(current, int) else f"different-{field}"})
    status = PipelineStatus(
        role=PeerRole.CLOUD,
        identity=remote_identity,
        sequence=1,
        session_id="session",
        session_generation=1,
        ready=True,
        runtime_state="ready",
    )

    update = edge.observe_cloud(status)

    assert update.error is not None
    assert update.error.code == code
    assert edge.state is PipelineState.HANDSHAKING


def test_heartbeat_loss_invalidates_all_in_flight_requests(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity, heartbeat_timeout=1.0)
    cloud = CloudSession(identity)
    _handshake(edge, cloud)
    edge.prepare_request(Operation.INFER, "first")
    edge.prepare_request(Operation.INFER, "second")

    update = edge.expire_heartbeat(now=11.1)

    assert update.error is not None
    assert update.error.code == "heartbeat_expired"
    assert update.invalidated_request_ids == ("first", "second")
    assert edge.state is PipelineState.DEGRADED


def test_heartbeat_recovery_requires_a_new_session(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity, heartbeat_timeout=1.0)
    cloud = CloudSession(identity)
    ready = _handshake(edge, cloud)

    edge.expire_heartbeat(now=11.1)
    stale = edge.observe_cloud(replace(ready, sequence=ready.sequence + 1), now=11.2)

    assert stale.error is not None
    assert stale.error.code == "stale_status"
    assert edge.state is PipelineState.DEGRADED

    assert cloud.observe_edge(edge.local_status(), backend_ready=True) is None
    recovered = cloud.status(
        backend_ready=True,
        backend_state="ready",
        reset_supported=False,
        cancellation_supported=False,
    )
    edge.observe_cloud(recovered, now=11.3)

    assert edge.ready
    assert recovered.session_id != ready.session_id
    assert recovered.session_generation > ready.session_generation


def test_cloud_restart_creates_new_session_and_stale_response_is_discarded(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    first_cloud = CloudSession(identity)
    first_status = _handshake(edge, first_cloud)
    old_request = edge.prepare_request(Operation.INFER, "old")

    restarted_cloud = CloudSession(identity)
    assert restarted_cloud.observe_edge(edge.local_status(), backend_ready=True) is None
    restarted_status = restarted_cloud.status(
        backend_ready=True,
        backend_state="ready",
        reset_supported=False,
        cancellation_supported=False,
    )
    update = edge.observe_cloud(restarted_status)

    assert update.invalidated_request_ids == ("old",)
    assert restarted_status.session_id != first_status.session_id

    edge.prepare_request(Operation.INFER, "current")
    stale = DistributedResult(
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
    stale_update = edge.accept_result(stale)

    assert stale_update.error is not None
    assert stale_update.error.code == "stale_response"


def test_cloud_session_rollover_waits_for_old_generation_runtime_operation(tmp_path):
    identity = _identity(tmp_path / "bundle")

    class Runtime:
        capabilities = SimpleNamespace(resettable=True, stateful=True, supports_cancellation=False)

        def __init__(self):
            self.infer_started = threading.Event()
            self.infer_release = threading.Event()
            self.active = 0
            self.max_active = 0
            self.policy_state = 0
            self.reset_calls = 0

        def infer(self, _request_id, _inputs, *, prompt=None, deadline=None):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.infer_started.set()
            try:
                assert self.infer_release.wait(timeout=2)
                self.policy_state += 1
                return SimpleNamespace(action=np.zeros((1, 6)), actual_chunk_size=1, backend_latency_ms=1.0)
            finally:
                self.active -= 1

        def reset(self, deadline=None):
            assert self.active == 0
            self.reset_calls += 1
            self.policy_state = 0

        @staticmethod
        def health():
            return SimpleNamespace(ready=True, state=SimpleNamespace(value="ready"))

        @staticmethod
        def close():
            return None

    runtime = Runtime()
    service = DistributedCloudService(identity, runtime)
    edge = EdgeSession(identity)
    edge.start()
    first_status = service.observe_edge(edge.local_status())
    assert runtime.reset_calls == 0
    edge.observe_cloud(first_status)
    old_request = edge.prepare_request(Operation.INFER, "old", inputs={})
    old_result: list[DistributedResult] = []
    old_thread = threading.Thread(target=lambda: old_result.append(service.handle(old_request)))
    old_thread.start()
    assert runtime.infer_started.wait(timeout=2)

    restarted_edge_status = replace(
        edge.local_status(),
        sequence=1,
        session_id="",
        session_generation=0,
        ready=False,
        runtime_state=PipelineState.HANDSHAKING.value,
    )
    rollover_status: list[PipelineStatus] = []
    rollover_thread = threading.Thread(
        target=lambda: rollover_status.append(service.observe_edge(restarted_edge_status))
    )
    rollover_thread.start()
    time.sleep(0.05)

    assert rollover_thread.is_alive()
    assert rollover_status == []
    runtime.infer_release.set()
    old_thread.join(timeout=2)
    rollover_thread.join(timeout=2)

    assert not old_thread.is_alive()
    assert not rollover_thread.is_alive()
    assert old_result[0].success is True
    assert rollover_status[0].session_id != first_status.session_id
    assert rollover_status[0].session_generation > first_status.session_generation
    assert runtime.max_active == 1
    assert runtime.reset_calls == 1
    assert runtime.policy_state == 0


def test_stateless_cloud_session_rollover_does_not_reset_runtime(tmp_path):
    identity = _identity(tmp_path / "bundle")

    class Runtime:
        capabilities = SimpleNamespace(resettable=False, stateful=False, supports_cancellation=False)

        def __init__(self):
            self.reset_calls = 0

        def reset(self, deadline=None):
            self.reset_calls += 1

        @staticmethod
        def health():
            return SimpleNamespace(ready=True, state=SimpleNamespace(value="ready"))

        @staticmethod
        def close():
            return None

    runtime = Runtime()
    service = DistributedCloudService(identity, runtime)
    edge = EdgeSession(identity)
    edge.start()
    first_status = service.observe_edge(edge.local_status())
    restarted = replace(edge.local_status(), sequence=1)

    rollover_status = service.observe_edge(restarted)

    assert rollover_status.session_id != first_status.session_id
    assert runtime.reset_calls == 0


def test_stateful_cloud_session_rollover_reset_failure_does_not_publish_new_session(tmp_path):
    identity = _identity(tmp_path / "bundle")

    class Runtime:
        capabilities = SimpleNamespace(resettable=True, stateful=True, supports_cancellation=False)

        def __init__(self):
            self.reset_calls = 0

        def reset(self, deadline=None):
            self.reset_calls += 1
            raise RuntimeError("reset failed")

        @staticmethod
        def health():
            return SimpleNamespace(ready=True, state=SimpleNamespace(value="ready"))

        @staticmethod
        def close():
            return None

    runtime = Runtime()
    service = DistributedCloudService(identity, runtime)
    edge = EdgeSession(identity)
    edge.start()
    service.observe_edge(edge.local_status())
    restarted = replace(edge.local_status(), sequence=1)

    failed_statuses = [service.observe_edge(restarted) for _ in range(3)]

    assert runtime.reset_calls == 3
    assert all(status.ready is False for status in failed_statuses)
    assert all(status.session_id == "" for status in failed_statuses)
    assert all(status.session_generation == 0 for status in failed_statuses)
    assert all(status.error is not None and status.error.stage == "reset" for status in failed_statuses)
    assert all("reset failed" in status.error.message for status in failed_statuses if status.error is not None)


def test_non_resettable_stateful_cloud_session_cannot_roll_over(tmp_path):
    identity = _identity(tmp_path / "bundle")

    class Runtime:
        capabilities = SimpleNamespace(resettable=False, stateful=True, supports_cancellation=False)

        @staticmethod
        def health():
            return SimpleNamespace(ready=True, state=SimpleNamespace(value="ready"))

        @staticmethod
        def close():
            return None

    service = DistributedCloudService(identity, Runtime())
    edge = EdgeSession(identity)
    edge.start()
    service.observe_edge(edge.local_status())
    restarted = replace(edge.local_status(), sequence=1)

    failed_statuses = [service.observe_edge(restarted) for _ in range(3)]

    assert all(status.ready is False for status in failed_statuses)
    assert all(status.session_id == "" for status in failed_statuses)
    assert all(status.error is not None for status in failed_statuses)
    assert all(
        status.error.code == "session_reset_unsupported" for status in failed_statuses if status.error is not None
    )


def test_remote_backend_failure_revokes_ready_session(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    cloud = CloudSession(identity)
    ready = _handshake(edge, cloud)
    edge.prepare_request(Operation.INFER, "request")
    failed = replace(
        ready,
        sequence=ready.sequence + 1,
        ready=False,
        runtime_state="failed",
        error=StructuredError(
            code="device_lost",
            message="cloud accelerator failed",
            stage="backend",
            recoverable=True,
        ),
    )

    update = edge.observe_cloud(failed)

    assert update.invalidated_request_ids == ("request",)
    assert update.error is not None
    assert update.error.code == "device_lost"
    assert edge.state is PipelineState.DEGRADED


def test_cloud_rejects_unknown_or_stale_session_requests(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    cloud = CloudSession(identity)
    _handshake(edge, cloud)
    request = edge.prepare_request(Operation.RESET, "reset")
    stale = replace(request, session_generation=request.session_generation + 1)

    with pytest.raises(DistributedProtocolError) as error:
        cloud.validate_request(stale)

    assert error.value.code == "session_generation_mismatch"


def test_cloud_rejects_replayed_request_ids(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    cloud = CloudSession(identity)
    _handshake(edge, cloud)
    request = edge.prepare_request(Operation.RESET, "reset")
    cloud.validate_request(request)

    with pytest.raises(DistributedProtocolError) as error:
        cloud.validate_request(request)

    assert error.value.code == "duplicate_request_id"


def test_cancel_request_requires_target_and_tracks_capability(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    cloud = CloudSession(identity)
    _handshake(edge, cloud)
    edge.prepare_request(Operation.INFER, "inference")

    cancel = edge.prepare_request(Operation.CANCEL, "cancel", target_request_id="inference")

    assert edge.cancellation_supported is True
    assert cancel.target_request_id == "inference"


def test_cancel_request_rejects_unknown_or_non_inference_target(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    cloud = CloudSession(identity)
    _handshake(edge, cloud)

    with pytest.raises(DistributedProtocolError) as unknown:
        edge.prepare_request(Operation.CANCEL, "cancel-unknown", target_request_id="missing")
    assert unknown.value.code == "invalid_cancel_target"

    edge.prepare_request(Operation.RESET, "reset")
    with pytest.raises(DistributedProtocolError) as non_inference:
        edge.prepare_request(Operation.CANCEL, "cancel-reset", target_request_id="reset")
    assert non_inference.value.code == "invalid_cancel_target"


def test_structured_cancellation_error_preserves_outcome_certainty():
    error = BackendCancellationError(
        "acknowledgement lost",
        operation_started=True,
        outcome_known=False,
    )

    structured = structured_error_from_exception(error, "cancel")

    assert structured.details["operation_started"] is True
    assert structured.details["outcome_known"] is False


@pytest.mark.parametrize("success", [True, False])
def test_cancel_result_must_match_requested_target(tmp_path, success):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    cloud = CloudSession(identity)
    status = _handshake(edge, cloud)
    edge.prepare_request(Operation.INFER, "target")
    cancel = edge.prepare_request(Operation.CANCEL, "cancel", target_request_id="target")
    result = DistributedResult(
        operation=Operation.CANCEL,
        pipeline_id=identity.pipeline_id,
        request_id=cancel.request_id,
        target_request_id="different-target",
        session_id=status.session_id,
        session_generation=status.session_generation,
        deployment_fingerprint=identity.deployment_fingerprint,
        success=success,
        error=None if success else StructuredError(code="cancel_failed", message="cancel failed", stage="cancel"),
        backend_ready=True,
        backend_state="ready",
    )

    update = edge.accept_result(result)

    assert update.error is not None
    assert update.error.code == "stale_response"
    assert update.canceled_request_id == ""


def test_expired_deadline_is_rejected_before_transmission(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    cloud = CloudSession(identity)
    _handshake(edge, cloud)

    with pytest.raises(DistributedProtocolError) as error:
        edge.prepare_request(
            Operation.INFER,
            "expired",
            deadline=datetime.now(timezone.utc) - timedelta(seconds=1),
        )

    assert error.value.code == "deadline_exceeded"


def test_structured_cloud_error_can_be_matched_without_action_payload(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    cloud = CloudSession(identity)
    status = _handshake(edge, cloud)
    request = edge.prepare_request(Operation.INFER, "failed")
    result = DistributedResult(
        operation=Operation.INFER,
        pipeline_id=identity.pipeline_id,
        request_id=request.request_id,
        session_id=status.session_id,
        session_generation=status.session_generation,
        deployment_fingerprint=identity.deployment_fingerprint,
        success=False,
        backend_ready=True,
        backend_state="ready",
        error=StructuredError(
            code="invalid_input",
            message="tensor decode failed",
            stage="decode",
        ),
    )

    update = edge.accept_result(result)

    assert update.error is None


def test_result_operation_must_match_pending_request(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    cloud = CloudSession(identity)
    status = _handshake(edge, cloud)
    edge.prepare_request(Operation.INFER, "request")
    mismatched = DistributedResult(
        operation=Operation.RESET,
        pipeline_id=identity.pipeline_id,
        request_id="request",
        session_id=status.session_id,
        session_generation=status.session_generation,
        deployment_fingerprint=identity.deployment_fingerprint,
        success=True,
        backend_ready=True,
        backend_state="ready",
    )

    update = edge.accept_result(mismatched)

    assert update.error is not None
    assert update.error.code == "stale_response"


def test_unknown_operation_error_can_complete_any_pending_operation(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    cloud = CloudSession(identity)
    status = _handshake(edge, cloud)
    edge.prepare_request(Operation.RESET, "request")
    failed = DistributedResult(
        operation=Operation.UNKNOWN,
        pipeline_id=identity.pipeline_id,
        request_id="request",
        session_id=status.session_id,
        session_generation=status.session_generation,
        deployment_fingerprint=identity.deployment_fingerprint,
        success=False,
        backend_ready=False,
        backend_state="decode_failed",
        error=StructuredError(
            code="decode_failed",
            message="invalid operation",
            stage="decode",
        ),
    )

    update = edge.accept_result(failed)

    assert update.error is not None
    assert update.error.code == "remote_backend_unavailable"


def test_dispatch_request_cannot_be_invalidated_between_registration_and_send(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    cloud = CloudSession(identity)
    status = _handshake(edge, cloud)
    observed: list[str] = []

    def sender(request):
        observed.append(request.request_id)
        failed = replace(
            status,
            sequence=status.sequence + 1,
            ready=False,
            runtime_state="failed",
            error=StructuredError(
                code="device_lost",
                message="cloud accelerator failed",
                stage="backend",
                recoverable=True,
            ),
        )
        update = edge.observe_cloud(failed)
        assert update.invalidated_request_ids == ("request",)

    request = edge.dispatch_request(Operation.INFER, "request", sender)

    assert request.request_id == "request"
    assert observed == ["request"]
    assert edge.state is PipelineState.DEGRADED


def test_cloud_startup_failure_is_reported_in_status_and_results(tmp_path):
    identity = _identity(tmp_path / "bundle")
    startup_error = StructuredError(
        code="missing_dependency",
        message="tcim runtime is unavailable",
        stage="startup",
    )
    service = DistributedCloudService(identity, None, startup_error=startup_error)
    edge = EdgeSession(identity)
    edge.start()

    status = service.observe_edge(edge.local_status())

    assert status.ready is False
    assert status.error == startup_error

    request = DistributedRequest(
        operation=Operation.RESET,
        pipeline_id=identity.pipeline_id,
        request_id="request",
        session_id="unavailable",
        session_generation=1,
        deployment_fingerprint=identity.deployment_fingerprint,
    )
    result = service.handle(request)

    assert result.success is False
    assert result.pipeline_id == identity.pipeline_id
    assert result.error == startup_error


def test_unknown_pipeline_error_uses_cloud_pipeline_identity(tmp_path):
    identity = _identity(tmp_path / "bundle")

    class Runtime:
        capabilities = type("Capabilities", (), {"resettable": False, "supports_cancellation": False})()

        @staticmethod
        def health():
            state = type("State", (), {"value": "ready"})()
            return type("Health", (), {"ready": True, "state": state})()

        @staticmethod
        def close():
            return None

    service = DistributedCloudService(identity, Runtime())
    edge = EdgeSession(identity)
    edge.start()
    cloud_status = service.observe_edge(edge.local_status())
    edge.observe_cloud(cloud_status)
    request = edge.prepare_request(Operation.RESET, "request")

    result = service.handle(replace(request, pipeline_id="unknown"))

    assert result.pipeline_id == identity.pipeline_id
    assert result.error is not None
    assert result.error.code == "pipeline_not_found"
    assert edge.accept_result(result).error is None


def test_cloud_status_advertises_reset_for_stateless_non_resettable_pipeline(tmp_path):
    identity = _identity(tmp_path / "bundle")

    class Runtime:
        capabilities = type(
            "Capabilities",
            (),
            {"resettable": False, "stateful": False, "supports_cancellation": False},
        )()

        @staticmethod
        def health():
            state = type("State", (), {"value": "ready"})()
            return type("Health", (), {"ready": True, "state": state})()

        @staticmethod
        def close():
            return None

    service = DistributedCloudService(identity, Runtime())
    edge = EdgeSession(identity)
    edge.start()

    status = service.observe_edge(edge.local_status())

    assert status.reset_supported is True


def test_cloud_status_preserves_session_while_pipeline_is_resetting(tmp_path):
    identity = _identity(tmp_path / "bundle")
    backend_health = type("BackendHealth", (), {"ready": True})()
    runtime = type(
        "Runtime",
        (),
        {
            "capabilities": type(
                "Capabilities",
                (),
                {"resettable": True, "stateful": True, "supports_cancellation": False},
            )(),
            "health": lambda _self: type(
                "Diagnostics",
                (),
                {
                    "ready": False,
                    "state": PipelineState.RESETTING,
                    "backend_health": backend_health,
                },
            )(),
            "close": lambda _self: None,
        },
    )()
    service = DistributedCloudService(identity, runtime)
    edge = EdgeSession(identity)
    edge.start()

    status = service.observe_edge(edge.local_status())

    assert status.ready is True
    assert status.runtime_state == "resetting"
    assert status.session_id


def test_ros_protocol_round_trips_status_request_and_result(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    cloud = CloudSession(identity)
    status = _handshake(edge, cloud)
    request = edge.prepare_request(
        Operation.INFER,
        "request",
        inputs={"observation.state": np.zeros((1, 6), dtype=np.float32)},
        deadline=datetime.now(timezone.utc) + timedelta(seconds=1),
        observation_timestamp_ns=1_000_000_000,
        stream_references=(StreamReference("observation.images.top", "top"),),
    )
    result = DistributedResult(
        operation=Operation.INFER,
        pipeline_id=identity.pipeline_id,
        request_id=request.request_id,
        session_id=status.session_id,
        session_generation=status.session_generation,
        deployment_fingerprint=identity.deployment_fingerprint,
        success=True,
        action=np.zeros((2, 6), dtype=np.float32),
        actual_chunk_size=2,
        backend_latency_ms=1.5,
        backend_ready=True,
        backend_state="ready",
    )

    assert status_from_message(status_to_message(status)) == status
    decoded_request = request_from_message(request_to_message(request))
    assert decoded_request.operation == request.operation
    assert decoded_request.request_id == request.request_id
    assert np.array_equal(decoded_request.inputs["observation.state"], request.inputs["observation.state"])
    assert decoded_request.observation_timestamp_ns == request.observation_timestamp_ns
    assert decoded_request.stream_references == request.stream_references
    decoded_result = result_from_message(result_to_message(result))
    assert decoded_result.actual_chunk_size == result.actual_chunk_size
    assert decoded_result.execution_horizon == result.execution_horizon
    assert np.array_equal(decoded_result.action, result.action)


def test_non_inference_operations_reject_stream_fields(tmp_path):
    identity = _identity(tmp_path / "bundle")

    with pytest.raises(ValueError, match="only for inference"):
        DistributedRequest(
            operation=Operation.RESET,
            pipeline_id=identity.pipeline_id,
            request_id="reset",
            session_id="session",
            session_generation=1,
            deployment_fingerprint=identity.deployment_fingerprint,
            observation_timestamp_ns=1,
            stream_references=(StreamReference("observation.images.top", "top"),),
        )


def test_stream_reference_cannot_collide_with_tensor_semantic(tmp_path):
    identity = _identity(tmp_path / "bundle")

    with pytest.raises(ValueError, match="collide"):
        DistributedRequest(
            operation=Operation.INFER,
            pipeline_id=identity.pipeline_id,
            request_id="request",
            session_id="session",
            session_generation=1,
            deployment_fingerprint=identity.deployment_fingerprint,
            inputs={"observation.images.top": np.zeros((1, 3, 4, 4), dtype=np.float32)},
            observation_timestamp_ns=1,
            stream_references=(StreamReference("observation.images.top", "top"),),
        )


def test_aligned_history_round_trips_through_the_request_message(tmp_path):
    identity = _identity(tmp_path / "bundle")
    request = DistributedRequest(
        operation=Operation.INFER,
        pipeline_id=identity.pipeline_id,
        request_id="request",
        session_id="session",
        session_generation=1,
        deployment_fingerprint=identity.deployment_fingerprint,
        observation_timestamp_ns=1,
        stream_references=(StreamReference("observation.images.top", "top"),),
        aligned_timestamps_ns=(1_000, 1_050, 1_100),
        aligned_tensors=(
            {"observation.state": np.array([0.0, 1.0])},
            {"observation.state": np.array([0.5, 1.5])},
            {"observation.state": np.array([1.0, 2.0])},
        ),
    )

    decoded = request_from_message(request_to_message(request))

    assert decoded.aligned_timestamps_ns == (1_000, 1_050, 1_100)
    assert len(decoded.aligned_tensors) == 3
    np.testing.assert_allclose(decoded.aligned_tensors[1]["observation.state"], np.array([0.5, 1.5]))


def test_aligned_history_requires_stream_references_and_equal_arrays(tmp_path):
    identity = _identity(tmp_path / "bundle")
    base = dict(
        operation=Operation.INFER,
        pipeline_id=identity.pipeline_id,
        request_id="request",
        session_id="session",
        session_generation=1,
        deployment_fingerprint=identity.deployment_fingerprint,
    )
    with pytest.raises(ValueError, match="equal length"):
        DistributedRequest(
            **base,
            stream_references=(StreamReference("observation.images.top", "top"),),
            aligned_timestamps_ns=(1_000,),
            aligned_tensors=(),
        )
    with pytest.raises(ValueError, match="requires stream references"):
        DistributedRequest(**base, aligned_timestamps_ns=(1_000,), aligned_tensors=({"k": 1},))


def test_align_inputs_to_selection_matches_the_nearest_history_entry():
    inputs = {"observation.state": [9.0, 9.0]}
    aligned, delta = align_inputs_to_selection(
        inputs,
        aligned_timestamps_ns=(1_000, 1_050, 1_100),
        aligned_tensors=(
            {"observation.state": [0.0, 0.0]},
            {"observation.state": [0.5, 0.5]},
            {"observation.state": [1.0, 1.0]},
        ),
        selected_capture_ns=1_060,
        tolerance_ns=25_000_000,
    )

    assert aligned["observation.state"] == [0.5, 0.5]
    assert delta == 10


def test_align_inputs_to_selection_fails_beyond_tolerance():
    with pytest.raises(DistributedProtocolError) as error:
        align_inputs_to_selection(
            {"observation.state": [9.0]},
            aligned_timestamps_ns=(1_000,),
            aligned_tensors=({"observation.state": [0.0]},),
            selected_capture_ns=1_000_000_000,
            tolerance_ns=25_000_000,
        )
    assert error.value.error.code == "state_alignment_unavailable"
    assert error.value.error.recoverable is True


def test_align_inputs_to_selection_skips_when_history_absent():
    inputs = {"observation.state": [9.0]}
    aligned, delta = align_inputs_to_selection(
        inputs,
        aligned_timestamps_ns=(),
        aligned_tensors=(),
        selected_capture_ns=1_000,
        tolerance_ns=25_000_000,
    )
    assert aligned is inputs
    assert delta == 0


def test_request_decoder_rejects_old_protocol_and_malformed_stream_arrays(tmp_path):
    identity = _identity(tmp_path / "bundle")
    request = DistributedRequest(
        operation=Operation.INFER,
        pipeline_id=identity.pipeline_id,
        request_id="request",
        session_id="session",
        session_generation=1,
        deployment_fingerprint=identity.deployment_fingerprint,
        observation_timestamp_ns=1,
        stream_references=(StreamReference("observation.images.top", "top"),),
    )
    message = request_to_message(request)
    message.protocol_version = PROTOCOL_VERSION - 1
    with pytest.raises(ValueError, match=f"expected {PROTOCOL_VERSION}"):
        request_from_message(message)

    message.protocol_version = PROTOCOL_VERSION
    message.stream_ids = []
    with pytest.raises(ValueError, match="equal length"):
        request_from_message(message)


def test_handshake_rejects_peer_from_previous_protocol_version(tmp_path):
    """A v(N-1) peer must fail the identity handshake, not enter READY.

    The result/action IDL gained ``execution_horizon`` in protocol v6, so a
    mixed-version deployment could otherwise pass the heartbeat identity
    check and then fail (or silently misdecode) on result exchange.
    """
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    edge.start()
    old_peer = replace(identity, protocol_version=PROTOCOL_VERSION - 1)
    status = PipelineStatus(
        role=PeerRole.CLOUD,
        identity=old_peer,
        sequence=1,
        session_id="session",
        session_generation=1,
        ready=True,
        runtime_state="ready",
    )

    update = edge.observe_cloud(status)

    assert update.error is not None
    assert update.error.code == "protocol_version_mismatch"
    assert update.error.details["remote"] == PROTOCOL_VERSION - 1
    assert edge.state is PipelineState.HANDSHAKING


def test_decode_failure_uses_cloud_identity_and_unknown_operation(tmp_path):
    identity = _identity(tmp_path / "bundle")
    message = request_to_message(
        DistributedRequest(
            operation=Operation.RESET,
            pipeline_id=identity.pipeline_id,
            request_id="request",
            session_id="session",
            session_generation=1,
            deployment_fingerprint=identity.deployment_fingerprint,
        )
    )
    message.operation = 255
    message.pipeline_id = "unknown"

    result = decode_failure_result(
        message,
        ValueError("invalid operation"),
        identity.deployment_fingerprint,
        identity.pipeline_id,
    )

    assert result.operation is Operation.UNKNOWN
    assert result.pipeline_id == identity.pipeline_id
    assert result.error is not None
    assert result.error.code == "decode_failed"


def test_build_aligned_history_decodes_raw_ros_messages():
    from sensor_msgs.msg import JointState

    from inference_service.pipeline_policy_node import PipelinePolicyNode
    from robot_config.contract_utils import StreamBuffer

    def joint_state(timestamp_ns, positions):
        message = JointState()
        message.header.stamp.sec = timestamp_ns // 1_000_000_000
        message.header.stamp.nanosec = timestamp_ns % 1_000_000_000
        message.name = [f"joint_{index}" for index in range(len(positions))]
        message.position = list(positions)
        return message

    buffer = StreamBuffer("hold", 1)
    base_ns = 1_000_000_000_000
    for index in range(5):
        timestamp_ns = base_ns + index * 20_000_000
        buffer.push(timestamp_ns, joint_state(timestamp_ns, [float(index)] * 3), receive_time_ns=timestamp_ns)

    spec = SimpleNamespace(key="observation.state", ros_type="sensor_msgs/msg/JointState", image_resize=None)
    host = SimpleNamespace(
        _state_alignment_window_ns=2_000_000_000,
        _state_specs=[spec],
        _subs={"observation.state": SimpleNamespace(spec=spec, buffer=buffer)},
        _joint_rad_limits={},
    )
    host._rad_to_lerobot = PipelinePolicyNode._rad_to_lerobot.__get__(host)

    timestamps, tensors = PipelinePolicyNode._build_aligned_history(
        host, {"observation.state": object()}, base_ns + 100_000_000
    )

    assert len(timestamps) == 5
    for index, entry in enumerate(tensors):
        value = entry["observation.state"]
        assert isinstance(value, np.ndarray)
        assert value.dtype == np.float32
        np.testing.assert_allclose(value, [float(index)] * 3)


def test_build_aligned_history_falls_back_with_multiple_state_sources():
    from inference_service.pipeline_policy_node import PipelinePolicyNode

    spec = SimpleNamespace(key="observation.state", ros_type="sensor_msgs/msg/JointState", image_resize=None)
    host = SimpleNamespace(
        _state_alignment_window_ns=2_000_000_000,
        _state_specs=[
            spec,
            SimpleNamespace(key="observation.gripper", ros_type="sensor_msgs/msg/JointState", image_resize=None),
        ],
        _subs={},
        get_logger=lambda: SimpleNamespace(warning=lambda *args, **kwargs: None),
    )

    timestamps, tensors = PipelinePolicyNode._build_aligned_history(
        host, {"observation.state": object()}, 1_000_000_000_000
    )

    assert timestamps == ()
    assert tensors == ()

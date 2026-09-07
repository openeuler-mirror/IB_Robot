"""Manifest-backed stateful Torch session for cumulative FullSubNet."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from inference_service.backends.errors import BackendInferenceError, BackendLoadError
from inference_service.backends.types import BackendAdmissionEvidence, BackendCapabilities, RuntimeContext
from inference_service.model_sessions.base import ModelSession
from inference_service.unified_runtime import ExecutionContext, LoadRollback, ModelRequest

from .contract import (
    FULLSUBNET_FB_FEATURES_SEMANTIC,
    FULLSUBNET_FB_SPECTRUM_SEMANTIC,
    FULLSUBNET_SB_FEATURES_SEMANTIC,
    FULLSUBNET_SB_MASK_SEMANTIC,
)
from .enhancement.fullsubnet_stateful_executor import (
    FB_FRAME_SHAPE,
    FB_OUTPUT_SHAPE,
    SB_FRAME_SHAPE,
    SB_OUTPUT_SHAPE,
)

_CHECKPOINT_ROLE = "fullsubnet_fb"
_STATE_CONTRACT_ROLE = "fullsubnet_sb"
_ROLE_ABI = {
    "fullsubnet_fb": (
        FULLSUBNET_FB_SPECTRUM_SEMANTIC,
        FULLSUBNET_FB_FEATURES_SEMANTIC,
        FB_FRAME_SHAPE,
        FB_OUTPUT_SHAPE,
    ),
    "fullsubnet_sb": (
        FULLSUBNET_SB_FEATURES_SEMANTIC,
        FULLSUBNET_SB_MASK_SEMANTIC,
        SB_FRAME_SHAPE,
        SB_OUTPUT_SHAPE,
    ),
}


class FullSubNetTorchSession(ModelSession):
    """Execute cumulative FullSubNet FB/SB roles from a Torch checkpoint deployment.

    The Torch deployment maps the shared cumulative checkpoint to the
    ``fullsubnet_fb`` artifact role and the cumulative contract manifest to the
    ``fullsubnet_sb`` artifact role; FB/SB recurrent state stays inside the
    executor and never crosses the host ABI.
    """

    allowed_runtime_options = frozenset({"timing_enabled"})

    def __init__(self) -> None:
        super().__init__(
            "model-session:fullsubnet-torch",
            BackendCapabilities(
                stateful=True,
                resettable=True,
                max_in_flight_per_instance=1,
                supports_multiple_instances=True,
                admission_evidence=BackendAdmissionEvidence(
                    sdk_initialization=True,
                    multi_instance_execution=True,
                    failure_isolation=True,
                    independent_close=True,
                ),
            ),
        )
        self._executor = None

    @property
    def backend(self) -> str:
        return self._executor.backend if self._executor is not None else "stateful_torch_cpu"

    @property
    def last_timing_ms(self) -> dict[str, float]:
        if self._executor is None:
            return {}
        return dict(self._executor.last_timing_ms)

    def _load(self, context: RuntimeContext, rollback: LoadRollback) -> None:
        if context.backend != "torch" or context.interface != "tensor_model":
            raise BackendLoadError(
                "FullSubNetTorchSession requires a tensor_model Torch deployment",
                code="invalid_deployment",
            )
        device = context.device
        if device not in {"cpu", "cuda"}:
            raise BackendLoadError(
                f"FullSubNetTorchSession requires a cpu/cuda Torch profile, got {device!r}",
                code="invalid_deployment",
            )
        unknown_options = sorted(set(context.runtime_options) - self.allowed_runtime_options)
        if unknown_options:
            raise BackendLoadError(
                f"unknown FullSubNet Torch session options: {unknown_options}",
                code="invalid_runtime_options",
            )
        artifacts = context.resolved_artifacts
        for role in (_CHECKPOINT_ROLE, _STATE_CONTRACT_ROLE):
            if role not in artifacts:
                raise BackendLoadError(
                    f"FullSubNet Torch deployment is missing artifact role {role!r}",
                    code="invalid_artifact",
                )
        timing_enabled = bool(context.runtime_options.get("timing_enabled", False))
        from .enhancement.fullsubnet_stateful_torch import StatefulTorchFullSubNetExecutor

        self._executor = StatefulTorchFullSubNetExecutor(
            str(artifacts[_CHECKPOINT_ROLE]),
            str(artifacts[_STATE_CONTRACT_ROLE]),
            device=device,
            timing_enabled=timing_enabled,
        )
        rollback.defer(self._release)

    def _execute(self, request: ModelRequest, context: ExecutionContext) -> Mapping[str, object]:
        context.check("backend")
        raise BackendInferenceError(
            "FullSubNet requires host-orchestrated role execution",
            code="host_orchestration_required",
        )

    def _execute_role(
        self,
        role: str,
        inputs: Mapping[str, object],
        request: ModelRequest,
        context: ExecutionContext,
    ) -> Mapping[str, object]:
        context.check(f"model.{role}")
        if self._executor is None:
            raise BackendInferenceError("FullSubNet Torch session is not loaded", code="runtime_not_loaded")
        abi = _ROLE_ABI.get(role)
        if abi is None:
            raise BackendInferenceError(f"unknown FullSubNet role {role!r}", code="unknown_execution_role")
        input_semantic, output_semantic, input_shape, output_shape = abi
        try:
            frame = np.asarray(inputs[input_semantic], dtype=np.float32)
        except KeyError as exc:
            raise BackendInferenceError(
                f"FullSubNet role {role!r} is missing semantic input {input_semantic!r}",
                code="missing_semantic_input",
            ) from exc
        if frame.shape != input_shape:
            raise BackendInferenceError(
                f"FullSubNet role {role!r} input shape {frame.shape} does not match {input_shape}",
                code="role_input_shape_mismatch",
            )
        try:
            output = self._executor.run_fb(frame) if role == "fullsubnet_fb" else self._executor.run_sb(frame)
        except Exception as exc:
            raise BackendInferenceError(
                f"FullSubNet role {role!r} failed and requires recovery: {exc}",
                code="state_outcome_unknown",
                recoverable=True,
                operation_started=True,
                outcome_known=False,
            ) from exc
        result = np.asarray(output, dtype=np.float32)
        if result.shape != output_shape:
            raise BackendInferenceError(
                f"FullSubNet role {role!r} output shape {result.shape} does not match {output_shape}",
                code="role_output_shape_mismatch",
            )
        return {output_semantic: result}

    def _reset(self) -> None:
        if self._executor is not None:
            self._executor.reset()

    def _close(self) -> None:
        if self._executor is not None:
            self._executor.close()
        self._release()

    def _release(self) -> None:
        self._executor = None


def build_fullsubnet_torch_session(context: RuntimeContext, *, providers=None) -> FullSubNetTorchSession:
    """Construct the manifest-selected FullSubNet Torch session without loading it."""

    del providers
    if context.backend != "torch" or context.model_type != "fullsubnet":
        raise BackendLoadError(
            "FullSubNet Torch session construction requires a tensor_model/fullsubnet Torch deployment",
            code="invalid_deployment",
        )
    return FullSubNetTorchSession()


__all__ = ["FullSubNetTorchSession", "build_fullsubnet_torch_session"]

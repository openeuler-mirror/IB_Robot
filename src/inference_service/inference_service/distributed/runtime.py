"""Split edge processor and cloud backend runtimes for distributed pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone

import numpy as np

from inference_manifest import ValidatedManifest
from inference_service.backends import (
    BackendCapabilities,
    BackendRegistry,
    InferenceRequest,
)
from inference_service.pipeline import create_pipeline_manager
from inference_service.pipeline.validation import validate_action_output
from inference_service.runtime_composition import require_runtime_dependencies
from inference_service.unified_runtime import RegistrySet, RuntimeProviders

from .types import UnsupportedDistributedRuntimeError


def _identity_preprocessor(inputs: Mapping[str, object]) -> Mapping[str, object]:
    return inputs


def _identity_postprocessor(action: object) -> object:
    return action


class EdgeProcessorRuntime:
    """Preserve the edge raw-observation and robot-unit boundary."""

    def __init__(
        self,
        pipeline_id: str,
        validated_manifest: ValidatedManifest,
        *,
        default_task: str | None = None,
    ) -> None:
        interface = getattr(getattr(validated_manifest.manifest, "model", None), "interface", None)
        if interface is not None and interface != "policy":
            raise UnsupportedDistributedRuntimeError(
                interface,
                getattr(validated_manifest.manifest.model, "model_type", ""),
            )
        self.pipeline_id = pipeline_id
        self._manifest = validated_manifest
        self._default_task = default_task
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True

    def preprocess(self, inputs: Mapping[str, object], *, prompt: str | None = None) -> Mapping[str, object]:
        self._require_ready()
        return dict(inputs)

    def postprocess(self, action: object, *, actual_chunk_size: int) -> object:
        self._require_ready()
        validate_action_output(
            action,
            actual_chunk_size=actual_chunk_size,
            action_dimension=self._manifest.policy.output_features["action"].shape[-1],
            pipeline_id=self.pipeline_id,
            phase="postprocessor",
        )
        return action

    def reset(self, deadline: datetime | None = None) -> None:
        self._require_ready()
        if deadline is not None and datetime.now(timezone.utc) >= deadline:
            raise TimeoutError("edge processor reset deadline expired")

    def close(self) -> None:
        self._loaded = False

    def _require_ready(self) -> None:
        if not self._loaded:
            raise RuntimeError("edge processors are not loaded")


class CloudBackendRuntime:
    """Execute raw edge observations with cloud-owned model processors."""

    def __init__(
        self,
        pipeline_id: str,
        validated_manifest: ValidatedManifest,
        *,
        request_timeout: float | None = None,
        runtime_options: Mapping[str, object] | None = None,
        registry: BackendRegistry | None = None,
        registry_set: RegistrySet | None = None,
        providers: RuntimeProviders | None = None,
    ) -> None:
        interface = getattr(getattr(validated_manifest.manifest, "model", None), "interface", None)
        if interface is not None and interface != "policy":
            raise UnsupportedDistributedRuntimeError(
                interface,
                getattr(validated_manifest.manifest.model, "model_type", ""),
            )
        registry_set, providers = require_runtime_dependencies(
            registry_set,
            providers,
            owner=type(self).__name__,
        )
        self.pipeline_id = pipeline_id
        self._manager = create_pipeline_manager(
            pipeline_id,
            validated_manifest,
            request_timeout=request_timeout,
            runtime_options=runtime_options,
            registry=registry,
            registry_set=registry_set,
            providers=providers,
            instrument_processors=True,
            model_component_id="cloud_inference",
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._manager.capabilities(self.pipeline_id)

    def infer(
        self,
        request_id: str,
        inputs: Mapping[str, object],
        *,
        prompt: str | None = None,
        deadline: datetime | None = None,
    ):
        import torch

        processor_inputs = {
            key: torch.from_numpy(np.ascontiguousarray(value)) if isinstance(value, np.ndarray) else value
            for key, value in inputs.items()
        }
        return self._manager.infer(
            self.pipeline_id,
            InferenceRequest(
                request_id=request_id,
                inputs=processor_inputs,
                prompt=prompt,
                deadline=deadline,
            ),
        )

    def reset(self, deadline: datetime | None = None) -> None:
        self._manager.reset(self.pipeline_id, deadline)

    def cancel(self, request_id: str, deadline: datetime | None = None) -> None:
        self._manager.cancel(self.pipeline_id, request_id, deadline)

    def health(self):
        return self._manager.health(self.pipeline_id)

    def close(self) -> None:
        self._manager.close()

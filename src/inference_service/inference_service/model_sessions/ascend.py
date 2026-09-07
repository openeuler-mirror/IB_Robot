"""Generic manifest-bound Ascend OM model session."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from inference_manifest import INTERNAL_SEMANTIC_PREFIX, CompiledDeployment, TensorBinding
from inference_service.backends.ascend.acl_runtime import (
    AclPriorityStreamPool,
    AclRuntimeLease,
    AclRuntimeManager,
)
from inference_service.backends.ascend.model import AclModel
from inference_service.backends.errors import (
    BackendAdmissionError,
    BackendInferenceError,
    BackendLifecycleError,
    BackendLoadError,
)
from inference_service.backends.types import (
    BackendAdmissionEvidence,
    BackendCapabilities,
    BackendPriorityMapping,
    BackendState,
    RuntimeContext,
)
from inference_service.model_sessions.base import ModelSession
from inference_service.scheduler.operations import Certainty
from inference_service.unified_runtime import ExecutionContext, LoadRollback, ModelRequest


class AscendOmModelSession(ModelSession):
    """Execute manifest-declared OM roles using shared ACL leases and model resources."""

    # Subclasses that need extra knobs (a denoising seed, say) widen this rather than
    # reimplementing _load; anything not listed is still rejected at load time.
    # Device placement is represented by AscendRuntimeProfile.  ``device_id``
    # remains a narrow source-level option for older pipeline callers and must
    # agree with the typed profile when both are present.
    allowed_runtime_options: frozenset[str] = frozenset({"device_id"})

    def __init__(
        self,
        device_id: int = 0,
        *,
        runtime_manager: AclRuntimeManager | None = None,
        model_factory=AclModel,
        priority_scheduling: bool = False,
        diagnostic_capture=None,
    ) -> None:
        if type(device_id) is not int or device_id < 0:
            raise ValueError("device_id must be a non-negative integer")
        if not isinstance(priority_scheduling, bool):
            raise TypeError("priority_scheduling must be a bool")
        if diagnostic_capture is not None and not callable(diagnostic_capture):
            raise TypeError("diagnostic_capture must be callable or None")
        if runtime_manager is None:
            raise BackendLoadError(
                "AscendOmModelSession requires an explicitly injected ACL runtime provider",
                code="acl_runtime_provider_required",
            )
        super().__init__(
            "model-session:ascend",
            BackendCapabilities(
                max_in_flight_per_instance=1,
                supports_multiple_instances=True,
                hardware_resource_id=f"ascend:{device_id}" if priority_scheduling else None,
                resource_domain=None if priority_scheduling else f"ascend:{device_id}",
                max_in_flight_per_resource_domain=None if priority_scheduling else 1,
                admission_evidence=BackendAdmissionEvidence(
                    sdk_initialization=True,
                    multi_instance_execution=True,
                    failure_isolation=True,
                    independent_close=True,
                ),
            ),
        )
        self._device_id = device_id
        self._priority_scheduling = priority_scheduling
        self._runtime_manager = runtime_manager
        self._model_factory = model_factory
        self._lease: AclRuntimeLease | None = None
        self._priority_streams: AclPriorityStreamPool | None = None
        self._models: dict[str, AclModel] = {}
        self._linked_inputs: frozenset[tuple[str, str]] = frozenset()
        self._diagnostic_capture = diagnostic_capture

    @property
    def runtime_version(self) -> str:
        return self._runtime_version(None if self._lease is None else self._lease.acl)

    def close(self) -> None:
        if not self._priority_scheduling:
            return super().close()
        with self._condition:
            if self._state in (BackendState.CLOSED, BackendState.CLOSING):
                return
            self._state = BackendState.CLOSING
        try:
            self._close()
        except Exception as exc:
            # Failed device drain retains ownership and permits a later close retry.
            with self._condition:
                self._failure_count += 1
                self._reason_code = getattr(exc, "code", "close_failed")
                self._message = str(exc)
                self._state = BackendState.FAILED
            raise
        with self._condition:
            self._context = None
            self._state = BackendState.CLOSED

    def drain_uncertain_operations(self) -> None:
        """排空所有阶段模型的隔离资源，不卸载模型。

        供上层（staged executor reset）在不关闭会话的前提下恢复设备侧
        确定性；任一阶段排空失败则抛出，保留剩余隔离资源以便重试。
        """
        for model in self._models.values():
            model.drain_quarantined()

    def _load(self, context: RuntimeContext, rollback: LoadRollback) -> None:
        if context.priority_scheduling != self._priority_scheduling:
            raise BackendLoadError(
                "Ascend model-session priority mode differs from its runtime context",
                code="deployment_context_mismatch",
            )
        deployment = context.deployment
        if not isinstance(deployment, CompiledDeployment) or context.backend != "ascend":
            raise BackendLoadError(
                "AscendOmModelSession requires a compiled Ascend deployment", code="invalid_deployment"
            )
        unknown_options = sorted(set(context.runtime_options) - self.allowed_runtime_options)
        if unknown_options:
            raise BackendLoadError(
                f"unknown Ascend model-session options: {unknown_options}", code="invalid_runtime_options"
            )
        profile_device_id = context.device_id
        device_id = context.runtime_options.get(
            "device_id", self._device_id if profile_device_id is None else profile_device_id
        )
        if type(device_id) is not int or device_id != self._device_id:
            raise BackendLoadError("Ascend device_id does not match the session", code="deployment_context_mismatch")
        if profile_device_id is not None and profile_device_id != self._device_id:
            raise BackendLoadError(
                "Ascend device_id does not match the typed runtime profile", code="deployment_context_mismatch"
            )
        if any(deployment.artifacts[role].format != "om" for role in deployment.execution):
            raise BackendLoadError("Ascend execution artifacts must use format 'om'", code="invalid_artifact_format")
        if context.target_runtime != "acl":
            raise BackendLoadError(
                f"Ascend target runtime {context.target_runtime!r} must be the canonical 'acl' family",
                code="incompatible_backend_target",
            )
        if any(link.producer_binding == "input" for link in deployment.device_links):
            raise BackendLoadError(
                "Ascend model sessions do not support input-sourced device links",
                code="unsupported_device_link_source",
            )
        lease = self._runtime_manager.acquire(self._device_id)
        rollback.defer(lease.close)
        priority_streams = AclPriorityStreamPool.create(lease) if self._priority_scheduling else None
        if priority_streams is not None:
            rollback.defer(priority_streams.close)
        models: dict[str, AclModel] = {}

        def close_models() -> None:
            for model in reversed(tuple(models.values())):
                model.close()

        rollback.defer(close_models)
        for role in deployment.execution:
            path = context.resolved_artifacts.get(role)
            if path is None or not path.is_file():
                raise BackendLoadError(f"Ascend artifact {role!r} is unavailable: {path}", code="invalid_artifact")
            model_options = {"async_execution": True} if self._priority_scheduling else {}
            model = self._model_factory(lease, role, path, deployment.bindings[role], **model_options)
            models[role] = model
            model.load_descriptor()
        linked_inputs = {(link.consumer, link.semantic) for link in deployment.device_links}
        self._prepare_models(deployment, models)
        self._lease = lease
        self._priority_streams = priority_streams
        self._models = models
        self._linked_inputs = frozenset(linked_inputs)
        self._update_loaded_capabilities(
            resettable=self.supports_reset,
            stateful=self.is_stateful,
            supports_attention=False,
            supports_isolated_stage_execution=priority_streams is not None,
            priority_mapping=(
                BackendPriorityMapping(tuple(range(priority_streams.level_count)))
                if priority_streams is not None
                else None
            ),
        )

    def _prepare_models(self, deployment: CompiledDeployment, models: Mapping[str, AclModel]) -> None:
        """Prepare role datasets after every model descriptor has been loaded."""

        for role in deployment.execution:
            input_overrides = {}
            for link in deployment.device_links:
                if link.consumer != role:
                    continue
                producer_binding = self._binding_for_semantic(
                    deployment.bindings[link.producer].outputs, link.semantic, "producer output"
                )
                consumer_binding = self._binding_for_semantic(
                    deployment.bindings[link.consumer].inputs, link.semantic, "consumer input"
                )
                if producer_binding.index is None or consumer_binding.index is None:
                    raise BackendLoadError(
                        f"Ascend device link {link.semantic!r} requires explicit runtime indices",
                        code="invalid_device_link",
                    )
                producer_buffer = models[link.producer].output_buffer(int(producer_binding.index))
                consumer_descriptor = models[role].input_descriptors[int(consumer_binding.index)]
                if producer_buffer.size != consumer_descriptor.size:
                    raise BackendLoadError(
                        f"Ascend device link {link.semantic!r} size differs between producer "
                        f"({producer_buffer.size}) and consumer ({consumer_descriptor.size})",
                        code="device_link_size_mismatch",
                    )
                input_overrides[int(consumer_binding.index)] = producer_buffer
            models[role].prepare_datasets(input_overrides=input_overrides)

    def _execute(self, request: ModelRequest, context: ExecutionContext) -> Mapping[str, object]:
        context.check("backend")
        deployment = self._loaded_deployment()
        values = dict(request.inputs)
        public_outputs: dict[str, object] = {}
        for role_index, role in enumerate(deployment.execution):
            for semantic, output in self._run_role(role_index, role, values, request=request, context=context).items():
                if not semantic.startswith(INTERNAL_SEMANTIC_PREFIX):
                    public_outputs[semantic] = output
        return public_outputs

    def _loaded_deployment(self) -> CompiledDeployment:
        deployment = self._require_context().deployment
        if not isinstance(deployment, CompiledDeployment) or not self._models:
            raise BackendInferenceError("Ascend OM model session is not loaded", code="runtime_not_loaded")
        return deployment

    def _run_role(
        self,
        role_index: int,
        role: str,
        values: dict[str, object],
        *,
        request: ModelRequest | None = None,
        context: ExecutionContext | None = None,
        isolated: bool = False,
        operation_factory=None,
    ) -> dict[str, np.ndarray]:
        """Execute one manifest role and write its outputs back into ``values``.

        Sessions whose roles are joined by the compiled graph alone walk ``execution``
        straight through, but a host-orchestrated model computes tensors between roles and
        drives them one at a time. Both need identical binding, device-link and
        read-selection behaviour per role, so that behaviour lives here rather than in the
        loop that happens to call it.
        """
        execute_options: dict[str, object] = {
            "read_outputs": (
                {
                    int(binding.index)
                    for binding in self._loaded_deployment().bindings[role].outputs
                    if binding.index is not None
                }
                if isolated
                else self._role_read_indices(role_index, role)
            ),
        }
        if request is not None:
            stream = self._request_stream(request, context)
            if stream is not None:
                execute_options["stream"] = stream
        role_inputs = self._role_inputs(role, values, isolated=isolated)
        self._capture_role_inputs(role, role_inputs)
        if isolated:
            runtime_outputs = self._submit_isolated_role(
                role, role_inputs, context, operation_factory=operation_factory, **execute_options
            )
        else:
            runtime_outputs = self._models[role].execute(role_inputs, **execute_options)
        outputs: dict[str, np.ndarray] = {}
        for binding in self._loaded_deployment().bindings[role].outputs:
            if binding.index is not None and int(binding.index) not in runtime_outputs:
                continue
            output = self._bound_output(role, binding, runtime_outputs)
            values[binding.semantic] = output
            outputs[binding.semantic] = output
        self._capture_role_outputs(role, outputs)
        if self._diagnostic_capture is not None and not isolated:
            linked_outputs = {
                link.semantic
                for link in self._loaded_deployment().device_links
                if link.producer == role and link.transport == "device_pointer"
            }
            outputs = {semantic: value for semantic, value in outputs.items() if semantic not in linked_outputs}
        return outputs

    def _capture_role_inputs(self, role: str, inputs: Mapping[int, np.ndarray]) -> None:
        if self._diagnostic_capture is None:
            return
        image_index = 0
        bindings = {int(binding.index): binding for binding in self._loaded_deployment().bindings[role].inputs}
        for index, value in sorted(inputs.items()):
            semantic = bindings[index].semantic
            if role == "vlm" and semantic.startswith("observation.images."):
                name = f"vlm_in_image_{image_index}"
                image_index += 1
            elif role == "vlm" and semantic == "observation.language.tokens":
                name = "vlm_in_lang_tokens"
            elif role == "vlm" and semantic == "observation.language.attention_mask":
                name = "vlm_in_lang_masks"
            elif role == "vlm" and semantic == "prefix_att_2d_masks_4d":
                name = "vlm_in_prefix_mask_4d"
            else:
                name = f"{role}_in_{semantic}"
            self._diagnostic_capture(name, value)

    def _capture_role_outputs(self, role: str, outputs: Mapping[str, object]) -> None:
        if self._diagnostic_capture is None:
            return
        names = {
            "internal.past_kv": "past_kv_tensor",
            "internal.prefix_pad_masks": "prefix_pad_masks",
        }
        for semantic, value in outputs.items():
            self._diagnostic_capture(names.get(semantic, f"{role}_out_{semantic}"), value)

    def _role_inputs(self, role: str, values: Mapping[str, object], *, isolated: bool = False) -> dict[int, np.ndarray]:
        """Gather one role's runtime inputs, skipping the slots a device link already fills."""
        indexed_inputs: dict[int, np.ndarray] = {}
        for binding in self._loaded_deployment().bindings[role].inputs:
            if binding.index is None:
                raise BackendInferenceError(
                    f"Ascend input {binding.semantic!r} has no runtime index", code="invalid_input_bindings"
                )
            if not isolated and (role, binding.semantic) in self._linked_inputs:
                continue
            try:
                indexed_inputs[int(binding.index)] = np.asarray(values[binding.semantic])
            except KeyError as exc:
                raise BackendInferenceError(
                    f"Ascend role {role!r} is missing semantic input {binding.semantic!r}",
                    code="missing_semantic_input",
                ) from exc
        return indexed_inputs

    def _role_read_indices(self, role_index: int, role: str) -> set[int]:
        """Select the output slots worth copying back to the host for one role.

        Everything that leaves the compiled graph is read, plus any internal tensor a later
        role consumes over something other than a device link - those still have to travel
        through host memory.
        """
        deployment = self._loaded_deployment()
        bindings = deployment.bindings[role]
        if self._diagnostic_capture is not None:
            return {int(binding.index) for binding in bindings.outputs if binding.index is not None}
        public_indices = {
            int(binding.index)
            for binding in bindings.outputs
            if binding.index is not None and not binding.semantic.startswith(INTERNAL_SEMANTIC_PREFIX)
        }
        host_internal_indices = {
            int(binding.index)
            for binding in bindings.outputs
            if binding.index is not None
            and binding.semantic.startswith(INTERNAL_SEMANTIC_PREFIX)
            and any(
                later_input.semantic == binding.semantic
                for later_role in deployment.execution[role_index + 1 :]
                for later_input in deployment.bindings[later_role].inputs
                if not any(
                    link.producer == role and link.consumer == later_role and link.semantic == binding.semantic
                    for link in deployment.device_links
                )
            )
        }
        return public_indices | host_internal_indices

    def _execute_role(
        self,
        role: str,
        inputs: Mapping[str, object],
        request: ModelRequest,
        context: ExecutionContext,
    ) -> Mapping[str, object]:
        deployment = self._loaded_deployment()
        try:
            role_index = deployment.execution.index(role)
        except ValueError as exc:
            raise BackendInferenceError(
                f"unknown or unloaded Ascend role {role!r}", code="unknown_execution_role"
            ) from exc
        return self._run_role(role_index, role, dict(inputs), request=request, context=context)

    def _execute_role_isolated(
        self,
        role: str,
        inputs: Mapping[str, object],
        request: ModelRequest,
        context: ExecutionContext,
        *,
        operation_factory=None,
    ) -> Mapping[str, object]:
        """Execute one role with request-owned datasets for staged overlap."""

        deployment = self._loaded_deployment()
        if role not in deployment.execution:
            raise BackendInferenceError(f"unknown Ascend stage {role!r}", code="unknown_execution_role")
        return self._run_role(
            deployment.execution.index(role),
            role,
            dict(inputs),
            request=request,
            context=context,
            isolated=True,
            operation_factory=operation_factory,
        )

    def _submit_isolated_role(self, role, role_inputs, context, *, read_outputs, stream=None, operation_factory=None):
        model = self._models[role]
        operation = registry = None
        if stream is not None and callable(operation_factory):
            operation, registry = operation_factory(role)
            if not operation.claim_send():
                registry.detach_waiter(operation.operation_id, context.request_id)
                raise BackendInferenceError(
                    f"stage operation {operation.operation_id} is already owned",
                    code="stage_operation_not_retryable",
                    operation_started=True,
                    outcome_known=False,
                )
        try:
            if stream is None:
                runtime_outputs = model.execute(role_inputs, read_outputs=read_outputs)
            else:
                submitted = model.submit_async_isolated(
                    role_inputs,
                    stream=stream,
                    read_outputs=read_outputs,
                )
                if operation is not None and registry is not None:
                    registry.bind_future(operation.operation_id, submitted.completion)
                runtime_outputs = submitted.completion.result()
                # Future.result() may wake before its registry callback runs.
                # Publish completion before an iteration reuses the key.
                if operation is not None and registry is not None:
                    registry.finish(operation.operation_id, certainty=Certainty.COMPLETED, result=runtime_outputs)
        except Exception as exc:
            if operation is not None and registry is not None:
                certainty = Certainty.UNKNOWN
                if bool(getattr(exc, "outcome_known", True)):
                    certainty = (
                        Certainty.COMPLETED if bool(getattr(exc, "operation_started", False)) else Certainty.NOT_STARTED
                    )
                registry.finish(operation.operation_id, certainty=certainty, error=str(exc))
            raise
        finally:
            if operation is not None and registry is not None:
                registry.detach_waiter(operation.operation_id, context.request_id)
        return runtime_outputs

    def _request_stream(self, request: ModelRequest, context: ExecutionContext | None = None) -> object | None:
        del context
        priority = request.metadata.get("priority", 0)
        streams = self._priority_streams
        if streams is None:
            if priority != 0:
                raise BackendAdmissionError(
                    "non-zero Ascend priority requires scheduler-enabled priority streams",
                    code="hardware_priority_unavailable",
                )
            return None
        mapping = self.capabilities.priority_mapping
        if mapping is None:
            raise BackendAdmissionError(
                "Ascend priority mapping is unavailable",
                code="hardware_priority_unavailable",
            )
        try:
            native_priority = mapping.map_generic(priority)
        except ValueError as exc:
            raise BackendAdmissionError(str(exc), code="unsupported_priority") from exc
        if not streams.supports(native_priority):
            raise BackendAdmissionError(
                f"Ascend hardware does not expose native priority {native_priority}",
                code="hardware_priority_unavailable",
            )
        return streams.select(native_priority)

    def _validate_request(self, request: ModelRequest, context: ExecutionContext) -> None:
        del context
        self._request_stream(request)

    def _close(self) -> None:
        models = self._models
        lease = self._lease
        priority_streams = self._priority_streams
        if not self._priority_scheduling:
            self._models = {}
            self._lease = None
            self._priority_streams = None
            self._linked_inputs = frozenset()
        errors: list[Exception] = []
        for model in reversed(tuple(models.values())):
            try:
                model.close()
            except Exception as exc:
                if self._priority_scheduling:
                    raise BackendLifecycleError(
                        f"Ascend model drain failed: {exc}", code="close_failed", close_pending=True
                    ) from exc
                errors.append(exc)
        self._models = {}
        if priority_streams is not None:
            try:
                priority_streams.close()
            except Exception as exc:
                errors.append(exc)
        if lease is not None:
            try:
                lease.close()
            except Exception as exc:
                errors.append(exc)
        self._lease = None
        self._priority_streams = None
        self._linked_inputs = frozenset()
        if errors:
            raise BackendLifecycleError(
                f"Ascend model session close failed: {'; '.join(str(error) for error in errors)}",
                code="close_failed",
            )

    @staticmethod
    def _bound_output(role: str, binding: TensorBinding, outputs: Mapping[int, np.ndarray]) -> np.ndarray:
        if binding.index is None or int(binding.index) not in outputs:
            raise BackendInferenceError(
                f"Ascend role {role!r} did not return output {binding.semantic!r}", code="missing_runtime_output"
            )
        return outputs[int(binding.index)]

    @staticmethod
    def _binding_for_semantic(bindings, semantic: str, description: str) -> TensorBinding:
        matches = [binding for binding in bindings if binding.semantic == semantic]
        if len(matches) != 1:
            raise BackendLoadError(
                f"Ascend device link {semantic!r} requires exactly one {description} binding",
                code="invalid_device_link",
            )
        return matches[0]


def build_ascend_model_session(
    context: RuntimeContext,
    *,
    providers=None,
) -> AscendOmModelSession:
    """Select the Ascend session execution mode from the deployment contract."""

    deployment = context.deployment
    if not isinstance(deployment, CompiledDeployment) or context.backend != "ascend":
        raise BackendLoadError(
            "Ascend session construction requires a compiled Ascend deployment", code="invalid_deployment"
        )
    if context.target_runtime != "acl":
        raise BackendLoadError(
            "Ascend session construction requires target.runtime='acl'",
            code="incompatible_backend_target",
        )
    device_id = context.device_id
    if device_id is None:
        device_id = context.runtime_options.get("device_id", 0)
    if type(device_id) is not int or device_id < 0:
        raise BackendLoadError("Ascend device_id must be a non-negative integer", code="invalid_runtime_options")
    session_type = AscendOmModelSession
    contract = deployment.execution_contract
    if contract.stateful or contract.state_links:
        from inference_service.model_sessions.ascend_stateful import StatefulAscendOmModelSession

        session_type = StatefulAscendOmModelSession
    return session_type(
        device_id=device_id,
        priority_scheduling=context.priority_scheduling,
        runtime_manager=(getattr(providers, "acl_runtime_provider", None) if providers is not None else None),
    )

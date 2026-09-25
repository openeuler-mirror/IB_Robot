"""Generic manifest-bound Houmo HMM (TCIM) model session."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence

from inference_manifest import CompiledDeployment, TensorBinding
from inference_service.backends.errors import BackendInferenceError, BackendLifecycleError, BackendLoadError
from inference_service.backends.hmm.backend import HMMModule
from inference_service.backends.types import BackendAdmissionEvidence, BackendCapabilities, RuntimeContext
from inference_service.model_sessions.base import ModelSession
from inference_service.unified_runtime import ExecutionContext, LoadRollback, ModelRequest

_HOST_ARTIFACT_FORMATS = frozenset({"pt", "pytorch"})


class HMMModelSession(ModelSession):
    """Execute manifest-declared TCIM roles through shared HMM module resources.

    The session owns only resource lifecycle (TCIM runtime, weight manager, role
    modules, device handles), admission, device-link setup, manifest-bound role
    invocation, and close. It deliberately does not implement PI0.5/SmolVLA
    timestep loops, Euler/state updates, embedding construction, time embedding,
    or noise sampling; those responsibilities belong to executor-owned stages
    above this session that drive it through ``execution().invoke(role, ...)``.
    """

    def __init__(
        self,
        device_id: int = 0,
        *,
        runtime_loader: Callable[[], object] | None = None,
    ) -> None:
        if type(device_id) is not int or device_id < 0:
            raise ValueError("device_id must be a non-negative integer")
        super().__init__(
            "model-session:hmm",
            BackendCapabilities(
                max_in_flight_per_instance=1,
                supports_multiple_instances=True,
                resource_domain=f"hmm:{device_id}",
                max_in_flight_per_resource_domain=1,
                admission_evidence=BackendAdmissionEvidence(
                    sdk_initialization=True,
                    multi_instance_execution=True,
                    failure_isolation=True,
                    independent_close=True,
                ),
            ),
        )
        self._device_id = device_id
        self._runtime_loader = runtime_loader
        self._runtime: object | None = None
        self._weight_manager: object | None = None
        self._modules: dict[str, HMMModule] = {}
        self._host_roles: frozenset[str] = frozenset()
        self._device_handles: tuple[object, ...] = ()

    @property
    def runtime_version(self) -> str:
        return self._runtime_version(self._runtime)

    def _load(self, context: RuntimeContext, rollback: LoadRollback) -> None:
        deployment = context.deployment
        if not isinstance(deployment, CompiledDeployment) or context.backend != "hmm":
            raise BackendLoadError("HMMModelSession requires a compiled hmm deployment", code="invalid_deployment")
        if context.target_runtime not in {"hmm", "tcim"}:
            raise BackendLoadError(
                f"HMM target.runtime {context.target_runtime!r} is not the canonical TCIM runtime family",
                code="incompatible_backend_target",
            )
        profile_device_id = context.device_id
        device_id = context.runtime_options.get(
            "device_id", self._device_id if profile_device_id is None else profile_device_id
        )
        if type(device_id) is not int or device_id != self._device_id:
            raise BackendLoadError("HMM device_id does not match the session", code="deployment_context_mismatch")
        if profile_device_id is not None and profile_device_id != self._device_id:
            raise BackendLoadError(
                "HMM device_id does not match the typed runtime profile", code="deployment_context_mismatch"
            )
        unknown_options = sorted(set(context.runtime_options) - {"device_id"})
        if unknown_options:
            raise BackendLoadError(
                f"unknown HMM model-session options: {unknown_options}", code="invalid_runtime_options"
            )

        runtime = self._import_runtime()
        weight_manager = self._create_weight_manager(runtime, self._device_id)
        modules: dict[str, HMMModule] = {}
        host_roles: list[str] = []

        def cleanup_resources() -> None:
            errors: list[Exception] = []
            for module in reversed(tuple(modules.values())):
                try:
                    module.close()
                except Exception as exc:
                    errors.append(exc)
            try:
                self._release_resource(weight_manager)
            except Exception as exc:
                errors.append(exc)
            if errors:
                raise RuntimeError("; ".join(str(error) for error in errors))

        rollback.defer(cleanup_resources)

        for role in deployment.execution:
            artifact = deployment.artifacts[role]
            if artifact.format in _HOST_ARTIFACT_FORMATS:
                host_roles.append(role)
                continue
            if artifact.format != "hmm":
                raise BackendLoadError(
                    f"HMM role {role!r} artifact format must be 'hmm', 'pt', or 'pytorch'",
                    code="invalid_artifact_format",
                )
            path = context.resolved_artifacts.get(role)
            if path is None or not path.is_file():
                raise BackendLoadError(f"HMM artifact {role!r} is unavailable: {path}", code="invalid_artifact")
            option = self._create_option(runtime, weight_manager)
            try:
                modules[role] = HMMModule(runtime, role, path, deployment.bindings[role], option=option)
            except BackendLoadError:
                raise
            except Exception as exc:
                raise BackendLoadError(
                    f"HMM role {role!r} failed to load from manifest artifact: {exc}",
                    code="runtime_load_failed",
                ) from exc

        handles: list[object] = []
        for link in deployment.device_links:
            source_bindings = self._device_link_source_bindings(deployment, link.producer, link.producer_binding)
            source = self._binding_for_semantic(source_bindings, link.semantic, "device-link source")
            target = self._binding_for_semantic(
                deployment.bindings[link.consumer].inputs,
                link.semantic,
                "device-link consumer",
            )
            handle = modules[link.producer].get_device_source(source, link.producer_binding)
            modules[link.consumer].set_device_input(target, handle)
            handles.append(handle)

        self._runtime = runtime
        self._weight_manager = weight_manager
        self._modules = modules
        self._host_roles = frozenset(host_roles)
        self._device_handles = tuple(handles)

    def _execute(self, request: ModelRequest, context: ExecutionContext) -> Mapping[str, object]:
        context.check("backend")
        context = self._require_context()
        deployment = context.deployment
        if not isinstance(deployment, CompiledDeployment) or not self._modules:
            raise BackendInferenceError("HMM model session is not loaded", code="runtime_not_loaded")
        values: dict[str, object] = dict(request.inputs)
        public_outputs: dict[str, object] = {}
        for role in deployment.execution:
            if role in self._host_roles:
                continue
            outputs = self._execute_role(role, values, request, context)
            values.update(outputs)
            for semantic, value in outputs.items():
                if not semantic.startswith("internal."):
                    public_outputs[semantic] = value
        return public_outputs

    def _execute_role(
        self,
        role: str,
        inputs: Mapping[str, object],
        request: ModelRequest,
        context: ExecutionContext,
    ) -> Mapping[str, object]:
        del request
        context.check(f"model.{role}")
        context = self._require_context()
        deployment = context.deployment
        if not isinstance(deployment, CompiledDeployment) or role not in self._modules:
            if role in self._host_roles:
                raise BackendInferenceError(
                    f"HMM role {role!r} is a host role and is not executable by the model session",
                    code="host_role_not_executable",
                )
            raise BackendInferenceError(f"unknown or unloaded HMM role {role!r}", code="unknown_execution_role")
        bindings = deployment.bindings[role]
        linked_inputs = self._linked_input_semantics(deployment, role)
        linked_outputs = self._linked_output_semantics(deployment, role)
        semantic_inputs: dict[str, object] = {}
        for binding in bindings.inputs:
            if binding.semantic in linked_inputs:
                continue
            try:
                semantic_inputs[binding.semantic] = inputs[binding.semantic]
            except KeyError as exc:
                raise BackendInferenceError(
                    f"HMM role {role!r} is missing semantic input {binding.semantic!r}",
                    code="missing_semantic_input",
                ) from exc
        read_semantics = {binding.semantic for binding in bindings.outputs if binding.semantic not in linked_outputs}
        runtime_outputs = self._modules[role].execute(
            semantic_inputs,
            device_input_semantics=linked_inputs,
            read_semantics=read_semantics,
        )
        return {
            binding.semantic: self._output_value(runtime_outputs, binding)
            for binding in bindings.outputs
            if binding.semantic in read_semantics
        }

    def _close(self) -> None:
        modules = self._modules
        weight_manager = self._weight_manager
        self._modules = {}
        self._weight_manager = None
        self._runtime = None
        self._host_roles = frozenset()
        self._device_handles = ()
        errors: list[Exception] = []
        for module in reversed(tuple(modules.values())):
            try:
                module.close()
            except Exception as exc:
                errors.append(exc)
        try:
            self._release_resource(weight_manager)
        except Exception as exc:
            errors.append(exc)
        if errors:
            raise BackendLifecycleError(
                f"HMM model session close failed: {'; '.join(str(error) for error in errors)}",
                code="close_failed",
            )

    def _validate_role_values(
        self,
        role: str,
        inputs: Mapping[str, object],
        outputs: Mapping[str, object],
        *,
        isolated: bool = False,
    ) -> None:
        context = self._require_context()
        deployment = context.deployment
        if not isinstance(deployment, CompiledDeployment):
            raise BackendInferenceError("role execution requires a compiled deployment", code="invalid_deployment")
        try:
            bindings = deployment.bindings[role]
        except KeyError as exc:
            raise BackendInferenceError(f"unknown execution role {role!r}", code="unknown_execution_role") from exc
        # Isolated role execution never reaches here on the HMM backend (the
        # base class raises BackendCapabilityError first); accept the flag to
        # match the ModelSession._validate_role_values signature that
        # execute_role always passes.
        del isolated
        linked_inputs = self._linked_input_semantics(deployment, role)
        linked_outputs = self._linked_output_semantics(deployment, role)
        host_inputs = tuple(binding for binding in bindings.inputs if binding.semantic not in linked_inputs)
        host_outputs = tuple(binding for binding in bindings.outputs if binding.semantic not in linked_outputs)
        self._validate_values(inputs, host_inputs, f"role_{role}_input")
        self._validate_values(outputs, host_outputs, f"role_{role}_output")

    @staticmethod
    def _device_link_source_bindings(
        deployment: CompiledDeployment,
        producer: str,
        producer_binding: str,
    ) -> tuple[TensorBinding, ...]:
        bindings = deployment.bindings[producer]
        return bindings.inputs if producer_binding == "input" else bindings.outputs

    @staticmethod
    def _linked_input_semantics(deployment: CompiledDeployment, role: str) -> set[str]:
        return {
            link.semantic
            for link in deployment.device_links
            if link.consumer == role or (link.producer == role and link.producer_binding == "input")
        }

    @staticmethod
    def _linked_output_semantics(deployment: CompiledDeployment, role: str) -> set[str]:
        return {
            link.semantic
            for link in deployment.device_links
            if link.producer == role and link.producer_binding == "output"
        }

    @staticmethod
    def _output_value(outputs: Mapping[object, object], binding: TensorBinding) -> object:
        if binding.runtime_name is not None and binding.runtime_name in outputs:
            return outputs[binding.runtime_name]
        if binding.index is not None and int(binding.index) in outputs:
            return outputs[int(binding.index)]
        raise BackendInferenceError(
            f"HMM runtime did not return output {binding.semantic!r}",
            code="missing_runtime_output",
        )

    @staticmethod
    def _binding_for_semantic(
        bindings: Sequence[TensorBinding],
        semantic: str,
        description: str,
    ) -> TensorBinding:
        matches = [binding for binding in bindings if binding.semantic == semantic]
        if len(matches) != 1:
            raise BackendLoadError(
                f"HMM device link {semantic!r} requires exactly one {description} binding",
                code="invalid_device_link",
            )
        return matches[0]

    def _import_runtime(self) -> object:
        if self._runtime_loader is not None:
            return self._runtime_loader()
        try:
            module = importlib.import_module("tcim_lite")
            return module.runtime
        except (ImportError, OSError, AttributeError) as exc:
            raise BackendLoadError(
                f"TCIM dependency 'tcim_lite.runtime' is unavailable: {exc}",
                code="missing_dependency",
            ) from exc

    @staticmethod
    def _create_weight_manager(runtime: object, device_id: int) -> object | None:
        constructor = getattr(runtime, "WeightManager", None)
        if not callable(constructor):
            return None
        try:
            return constructor(device=device_id)
        except Exception as exc:
            raise BackendLoadError(
                f"Unable to create TCIM WeightManager for device {device_id}: {exc}",
                code="runtime_load_failed",
            ) from exc

    @staticmethod
    def _create_option(runtime: object, weight_manager: object | None) -> object | None:
        constructor = getattr(runtime, "Option", None)
        if weight_manager is None or not callable(constructor):
            return None
        return constructor(weight_manager)

    @staticmethod
    def _release_resource(resource: object | None) -> None:
        if resource is None:
            return
        for method_name in ("release", "close", "destroy"):
            method = getattr(resource, method_name, None)
            if callable(method):
                method()
                return

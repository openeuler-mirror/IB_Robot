"""Generic manifest-bound ONNX Runtime model session."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from inference_manifest import (
    INTERNAL_SEMANTIC_PREFIX,
    CompiledDeployment,
    ONNXRuntimeProfile,
    TensorBinding,
)
from inference_service.backends.errors import (
    BackendError,
    BackendInferenceError,
    BackendLoadError,
)
from inference_service.backends.types import (
    BackendAdmissionEvidence,
    BackendCapabilities,
    RuntimeContext,
)
from inference_service.model_sessions.base import ModelSession
from inference_service.unified_runtime import ExecutionContext, LoadRollback, ModelRequest

_CANONICAL_PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
}
_GRAPH_OPTIMIZATION_LEVELS = ("disable_all", "basic", "extended", "all")


def _import_onnxruntime() -> Any:
    try:
        return importlib.import_module("onnxruntime")
    except (ImportError, OSError) as exc:
        raise BackendLoadError(
            f"ONNX Runtime dependency 'onnxruntime' is unavailable: {exc}", code="missing_dependency"
        ) from exc


class OnnxRuntimeModelSession(ModelSession):
    """Execute manifest-declared ONNX graphs through explicit host tensor bindings.

    Each compiled execution role maps its semantic tensor contract onto ONNX
    graph inputs and outputs by ``runtime_name`` (or by binding index when the
    manifest omits the name).  Roles run sequentially in host memory: every
    public output of one role is available to later roles by semantic name.

    Recurrent roles follow the v3 state-link ABI (``<semantic>_<kind>_in`` /
    ``<semantic>_<kind>_out`` pairs).  State tensors live in host banks owned
    by this session, are excluded from the public host ABI, and are zeroed by
    ``reset()``.
    """

    allowed_runtime_options: frozenset[str] = frozenset()

    def __init__(self, *, stateful: bool = False, ort_loader: Callable[[], Any] | None = None) -> None:
        if not isinstance(stateful, bool):
            raise TypeError("stateful must be a bool")
        super().__init__(
            "model-session:onnx",
            BackendCapabilities(
                stateful=stateful,
                resettable=stateful,
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
        self._stateful = stateful
        self._ort_loader = ort_loader or _import_onnxruntime
        self._ort: Any | None = None
        self._sessions: dict[str, Any] = {}
        self._input_names: dict[str, dict[str, str]] = {}
        self._output_names: dict[str, dict[str, str]] = {}
        self._state_outputs: dict[str, dict[str, TensorBinding]] = {}
        self._state_input_semantics: dict[str, frozenset[str]] = {}
        self._state_banks: dict[str, dict[str, np.ndarray]] = {}

    @property
    def runtime_version(self) -> str:
        return self._runtime_version(self._ort)

    def _load(self, context: RuntimeContext, rollback: LoadRollback) -> None:
        profile = context.backend_profile
        deployment = context.deployment
        if context.backend != "onnx" or not isinstance(profile, ONNXRuntimeProfile):
            raise BackendLoadError(
                "OnnxRuntimeModelSession requires a tensor_model deployment with a typed ONNX Runtime profile",
                code="invalid_deployment",
            )
        if not isinstance(deployment, CompiledDeployment):
            raise BackendLoadError("OnnxRuntimeModelSession requires a compiled deployment", code="invalid_deployment")
        if context.target_runtime != "onnx":
            raise BackendLoadError(
                f"ONNX Runtime target runtime {context.target_runtime!r} must be 'onnx'",
                code="incompatible_backend_target",
            )
        unknown_options = sorted(context.runtime_options)
        if unknown_options:
            raise BackendLoadError(
                f"unknown ONNX Runtime model-session options: {unknown_options}", code="invalid_runtime_options"
            )
        if deployment.device_links:
            raise BackendLoadError(
                "ONNX Runtime sessions exchange tensors through host memory and do not support device links",
                code="unsupported_device_link_source",
            )
        contract = deployment.execution_contract
        state_links = contract.state_links
        if self._stateful != bool(contract.stateful or state_links):
            raise BackendLoadError(
                "ONNX Runtime model-session state mode differs from its deployment contract",
                code="deployment_context_mismatch",
            )
        if self._stateful and not state_links:
            raise BackendLoadError(
                "stateful ONNX Runtime sessions require manifest state_links", code="invalid_state_contract"
            )

        ort = self._ort_loader()
        providers = self._providers(ort, profile)
        sess_options = self._session_options(ort, profile)
        state_outputs = self._resolve_state_bindings(deployment, state_links)
        state_input_semantics = {role: frozenset(pairs) for role, pairs in state_outputs.items()}
        sessions: dict[str, Any] = {}
        input_names: dict[str, dict[str, str]] = {}
        output_names: dict[str, dict[str, str]] = {}
        state_banks: dict[str, dict[str, np.ndarray]] = {}

        def close_sessions() -> None:
            sessions.clear()

        rollback.defer(close_sessions)
        for role in deployment.execution:
            artifact = deployment.artifacts.get(role)
            if artifact is None or artifact.format != "onnx":
                raise BackendLoadError(
                    f"ONNX Runtime execution role {role!r} must use an 'onnx' artifact", code="invalid_artifact_format"
                )
            path = context.resolved_artifacts.get(role)
            if path is None or not path.is_file():
                raise BackendLoadError(
                    f"ONNX Runtime artifact {role!r} is unavailable: {path}", code="invalid_artifact"
                )
            try:
                session = ort.InferenceSession(str(path), sess_options=sess_options, providers=providers)
            except BackendError:
                raise
            except Exception as exc:
                raise BackendLoadError(f"unable to load ONNX graph for role {role!r}: {exc}") from exc
            sessions[role] = session
            graph_inputs = [item.name for item in session.get_inputs()]
            graph_outputs = [item.name for item in session.get_outputs()]
            bindings = deployment.bindings[role]
            input_names[role] = self._resolve_runtime_names(role, bindings.inputs, graph_inputs, "input")
            output_names[role] = self._resolve_runtime_names(role, bindings.outputs, graph_outputs, "output")
            unbound_inputs = sorted(set(graph_inputs) - set(input_names[role].values()))
            if unbound_inputs:
                raise BackendLoadError(
                    f"ONNX graph for role {role!r} has unbound inputs: {unbound_inputs}",
                    code="invalid_input_bindings",
                )
            bank = {}
            for input_semantic, output_binding in state_outputs.get(role, {}).items():
                bank[input_semantic] = np.zeros(output_binding.shape, dtype=np.dtype(output_binding.dtype))
            if bank:
                state_banks[role] = bank

        self._ort = ort
        self._sessions = sessions
        self._input_names = input_names
        self._output_names = output_names
        self._state_outputs = state_outputs
        self._state_input_semantics = state_input_semantics
        self._state_banks = state_banks
        rollback.defer(self._release)

    def _execute(self, request: ModelRequest, context: ExecutionContext) -> Mapping[str, object]:
        context.check("backend")
        deployment = self._loaded_deployment()
        values = dict(request.inputs)
        public_outputs: dict[str, object] = {}
        for role in deployment.execution:
            outputs = self._run_role(role, values, context)
            values.update(outputs)
            for semantic, output in outputs.items():
                if not semantic.startswith(INTERNAL_SEMANTIC_PREFIX):
                    public_outputs[semantic] = output
        return public_outputs

    def _execute_role(
        self,
        role: str,
        inputs: Mapping[str, object],
        request: ModelRequest,
        context: ExecutionContext,
    ) -> Mapping[str, object]:
        del request
        return self._run_role(role, dict(inputs), context)

    def _run_role(self, role: str, values: dict[str, object], context: ExecutionContext) -> dict[str, object]:
        context.check(f"model.{role}")
        deployment = self._loaded_deployment()
        session = self._sessions[role]
        semantic_inputs = self._input_names[role]
        state_bank = self._state_banks.get(role, {})
        feed: dict[str, np.ndarray] = {}
        for binding in deployment.bindings[role].inputs:
            semantic = binding.semantic
            runtime_name = semantic_inputs[semantic]
            if semantic in state_bank:
                feed[runtime_name] = state_bank[semantic]
                continue
            try:
                feed[runtime_name] = np.asarray(values[semantic])
            except KeyError as exc:
                raise BackendInferenceError(
                    f"ONNX Runtime role {role!r} is missing semantic input {semantic!r}",
                    code="missing_semantic_input",
                ) from exc
        try:
            runtime_outputs = session.run(None, feed)
        except BackendInferenceError:
            raise
        except Exception as exc:
            raise BackendInferenceError(
                f"ONNX Runtime role {role!r} failed: {exc}", code="onnxruntime_execute_failed"
            ) from exc
        output_names = self._output_names[role]
        reverse_outputs = {name: semantic for semantic, name in output_names.items()}
        outputs: dict[str, object] = {}
        for index, name in enumerate(_graph_output_names(session)):
            semantic = reverse_outputs.get(name)
            if semantic is None:
                raise BackendInferenceError(
                    f"ONNX Runtime role {role!r} returned unbound output {name!r}", code="missing_runtime_output"
                )
            outputs[semantic] = runtime_outputs[index]
        for input_semantic, output_binding in self._state_outputs.get(role, {}).items():
            state_bank[input_semantic] = np.asarray(outputs.pop(output_binding.semantic))
        return outputs

    def _reset(self) -> None:
        for role, bank in self._state_banks.items():
            for semantic, state_binding in self._state_outputs.get(role, {}).items():
                bank[semantic] = np.zeros(state_binding.shape, dtype=np.dtype(state_binding.dtype))

    def _close(self) -> None:
        self._release()

    def _release(self) -> None:
        self._sessions = {}
        self._input_names = {}
        self._output_names = {}
        self._state_outputs = {}
        self._state_input_semantics = {}
        self._state_banks = {}

    def _loaded_deployment(self) -> CompiledDeployment:
        deployment = self._require_context().deployment
        if not isinstance(deployment, CompiledDeployment) or not self._sessions:
            raise BackendInferenceError("ONNX Runtime model session is not loaded", code="runtime_not_loaded")
        return deployment

    def _providers(self, ort: Any, profile: ONNXRuntimeProfile) -> list[str]:
        requested = (profile.provider or profile.device or "cpu").strip()
        provider = _CANONICAL_PROVIDERS.get(requested.lower(), requested)
        if not provider.endswith("ExecutionProvider"):
            raise BackendLoadError(
                f"ONNX Runtime profile has unknown execution provider {requested!r}",
                code="invalid_runtime_profile",
            )
        try:
            available = list(ort.get_available_providers())
        except Exception as exc:
            raise BackendLoadError(f"unable to query ONNX Runtime providers: {exc}") from exc
        if provider not in available:
            raise BackendLoadError(
                f"ONNX Runtime execution provider {provider!r} is unavailable on this host",
                code="device_unavailable",
            )
        return [provider]

    def _session_options(self, ort: Any, profile: ONNXRuntimeProfile) -> Any:
        options = ort.SessionOptions()
        level = profile.optimization_level
        if level is None:
            return options
        normalized = level.strip().lower()
        if normalized not in _GRAPH_OPTIMIZATION_LEVELS:
            raise BackendLoadError(
                f"ONNX Runtime optimization level {level!r} must be one of {list(_GRAPH_OPTIMIZATION_LEVELS)}",
                code="invalid_runtime_profile",
            )
        levels = ort.GraphOptimizationLevel
        options.graph_optimization_level = {
            "disable_all": levels.ORT_DISABLE_ALL,
            "basic": levels.ORT_ENABLE_BASIC,
            "extended": levels.ORT_ENABLE_EXTENDED,
            "all": levels.ORT_ENABLE_ALL,
        }[normalized]
        return options

    def _resolve_state_bindings(
        self,
        deployment: CompiledDeployment,
        state_links: tuple[Any, ...],
    ) -> dict[str, dict[str, TensorBinding]]:
        """Resolve recurrent ABI pairs from logical v3 state links."""

        pairs: dict[str, dict[str, TensorBinding]] = {}
        for link in state_links:
            role = link.role
            if role == "__runtime__":
                role = link.state_bank.removesuffix(".bank")
            if role not in deployment.bindings:
                raise BackendLoadError(f"state link references undeclared role {role!r}", code="invalid_state_contract")
            state_kind = link.state_name
            bindings = deployment.bindings[role]
            input_matches = [
                binding
                for binding in bindings.inputs
                if binding.semantic.rsplit(".", 1)[-1].endswith(f"_{state_kind}_in")
            ]
            output_matches = [
                binding
                for binding in bindings.outputs
                if binding.semantic.rsplit(".", 1)[-1].endswith(f"_{state_kind}_out")
            ]
            if state_kind == "hidden" and not input_matches:
                input_matches = [binding for binding in bindings.inputs if binding.semantic.endswith(".state_in")]
                output_matches = [binding for binding in bindings.outputs if binding.semantic.endswith(".state_out")]
            if len(input_matches) != 1 or len(output_matches) != 1:
                raise BackendLoadError(
                    f"state link for role {role!r} has no unique ABI mapping for {state_kind!r}",
                    code="invalid_state_link_abi",
                )
            if input_matches[0].shape != output_matches[0].shape or input_matches[0].dtype != output_matches[0].dtype:
                raise BackendLoadError(
                    f"state link for role {role!r} changes shape or dtype across inference",
                    code="state_size_mismatch",
                )
            if any(dimension < 1 for dimension in input_matches[0].shape):
                raise BackendLoadError(
                    f"state link for role {role!r} requires a static shape, got {input_matches[0].shape}",
                    code="invalid_state_link_abi",
                )
            pairs.setdefault(role, {})[input_matches[0].semantic] = output_matches[0]
        return pairs

    @staticmethod
    def _resolve_runtime_names(
        role: str,
        bindings: tuple[TensorBinding, ...],
        graph_names: list[str],
        direction: str,
    ) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for binding in bindings:
            if binding.runtime_name is not None:
                name = binding.runtime_name
            elif binding.index is not None and binding.index < len(graph_names):
                name = graph_names[binding.index]
            else:
                raise BackendLoadError(
                    f"ONNX Runtime {direction} {binding.semantic!r} for role {role!r} has no graph slot",
                    code=f"invalid_{direction}_bindings",
                )
            if name not in graph_names:
                raise BackendLoadError(
                    f"ONNX graph for role {role!r} has no {direction} named {name!r} (semantic {binding.semantic!r})",
                    code=f"invalid_{direction}_bindings",
                )
            resolved[binding.semantic] = name
        return resolved

    def _validate_role_values(
        self,
        role: str,
        inputs: Mapping[str, object],
        outputs: Mapping[str, object],
    ) -> None:
        deployment = self._require_context().deployment
        if not isinstance(deployment, CompiledDeployment):
            raise BackendInferenceError("role execution requires a compiled deployment", code="invalid_deployment")
        bindings = deployment.bindings[role]
        state_inputs = self._state_input_semantics.get(role, frozenset())
        state_outputs = {binding.semantic for binding in self._state_outputs.get(role, {}).values()}
        host_inputs = tuple(binding for binding in bindings.inputs if binding.semantic not in state_inputs)
        host_outputs = tuple(binding for binding in bindings.outputs if binding.semantic not in state_outputs)
        self._validate_values(inputs, host_inputs, f"role_{role}_input")
        self._validate_values(outputs, host_outputs, f"role_{role}_output")


def _graph_output_names(session: Any) -> list[str]:
    return [item.name for item in session.get_outputs()]


def build_onnx_model_session(
    context: RuntimeContext,
    *,
    providers=None,
    ort_loader: Callable[[], Any] | None = None,
) -> OnnxRuntimeModelSession:
    """Select the ONNX Runtime session state mode from the deployment contract."""

    deployment = context.deployment
    if not isinstance(deployment, CompiledDeployment) or context.backend != "onnx":
        raise BackendLoadError(
            "ONNX Runtime session construction requires a compiled onnx deployment", code="invalid_deployment"
        )
    if context.target_runtime != "onnx":
        raise BackendLoadError(
            "ONNX Runtime session construction requires target.runtime='onnx'",
            code="incompatible_backend_target",
        )
    del providers  # ONNX Runtime owns its execution providers; no process provider is required.
    contract = deployment.execution_contract
    return OnnxRuntimeModelSession(
        stateful=bool(contract.stateful or contract.state_links),
        ort_loader=ort_loader,
    )


__all__ = ["OnnxRuntimeModelSession", "build_onnx_model_session"]

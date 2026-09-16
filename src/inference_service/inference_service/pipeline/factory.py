"""Unified construction of validated backend and pipeline instances."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from inference_manifest import (
    AscendRuntimeProfile,
    CompiledDeployment,
    HisiliconRuntimeProfile,
    HMMRuntimeProfile,
    RKNNRuntimeProfile,
    TensorBinding,
    TorchDeployment,
    TorchRuntimeProfile,
    ValidatedManifest,
)
from inference_manifest.json_utils import load_json_strict
from inference_service.backends import BackendLoadError, BackendRegistry, RuntimeContext
from inference_service.backends.rknn.runtime import validate_runtime_options as validate_rknn_runtime_options
from inference_service.codecs import build_execution_plan, create_policy_codec
from inference_service.model_sessions import (
    AscendOmModelSession,
    HisiliconModelSession,
    HMMModelSession,
    LeRobotTorchModelSession,
    RKNNModelSession,
)
from inference_service.model_sessions.lerobot_torch import LeRobotSessionPostprocessor, LeRobotSessionPreprocessor
from inference_service.pi05_schedule import PI05DenoisingSchedule, load_pi05_schedule, uniform_pi05_schedule
from inference_service.pipeline.executor import SequentialModelExecutor
from inference_service.pipeline.manager import InferencePipelineManager
from inference_service.pipeline.pi05 import create_pi05_executor, derive_pi05_topology
from inference_service.pipeline.pi05_hmm import (
    PI05HMMFamilyResource,
    build_embedding_stage,
    build_time_prep_stage,
    load_pi05_policy_config,
)
from inference_service.pipeline.processors import create_lerobot_processor_views
from inference_service.pipeline.runtime import InferencePipeline, _PolicySessionHandle, _RawActionResultAdapter
from inference_service.pipeline.smolvla import (
    SmolVLAFamilyResource,
    create_smolvla_executor,
    derive_smolvla_topology,
    load_smolvla_policy_config,
)
from inference_service.pipeline.stages import ModelStage
from inference_service.unified_runtime import (
    ModelRuntimeKey,
    RegistrySet,
    RuntimeAssemblerRegistry,
    RuntimeAssembly,
    RuntimeDependencyError,
    RuntimeDescriptor,
    RuntimeProviders,
    SessionBuilderKey,
)

_ALLOWED_ASCEND_OPTIONS = frozenset({"device_id"})
_ALLOWED_PI05_OPTIONS = _ALLOWED_ASCEND_OPTIONS | frozenset({"random_seed", "curvature_log_path"})
_ALLOWED_HMM_OPTIONS = frozenset({"device_id", "random_seed"})
_ALLOWED_RKNN_OPTIONS = frozenset({"target", "core_mask", "random_seed"})


def _new_lerobot_torch_session(
    context: RuntimeContext,
    *,
    options: Mapping[str, object],
    providers: RuntimeProviders | None = None,
) -> LeRobotTorchModelSession:
    del options
    deployment = context.deployment
    if not isinstance(deployment, TorchDeployment):
        raise BackendLoadError("LeRobot Torch policy requires a Torch deployment", code="invalid_deployment")
    if context.device is None:
        raise BackendLoadError("LeRobot Torch policy requires a typed Torch runtime profile", code="invalid_deployment")
    return LeRobotTorchModelSession(
        context.device,
        priority_scheduling=context.priority_scheduling,
    )


def _new_ascend_session(
    context: RuntimeContext,
    *,
    options: Mapping[str, object],
    providers: RuntimeProviders | None = None,
) -> AscendOmModelSession:
    device_id = context.device_id
    if device_id is None:
        device_id = options.get("device_id", 0)
    return AscendOmModelSession(
        device_id=int(device_id),
        priority_scheduling=context.priority_scheduling,
        runtime_manager=(getattr(providers, "acl_runtime_provider", None) if providers is not None else None),
    )


def _new_hmm_session(
    context: RuntimeContext,
    *,
    options: Mapping[str, object],
    providers: RuntimeProviders | None = None,
) -> HMMModelSession:
    device_id = context.device_id
    if device_id is None:
        device_id = options.get("device_id", 0)
    return HMMModelSession(
        device_id=int(device_id),
    )


def _new_rknn_session(
    context: RuntimeContext,
    *,
    options: Mapping[str, object],
    providers: RuntimeProviders | None = None,
) -> RKNNModelSession:
    del context, options
    return RKNNModelSession()


def _new_hisilicon_session(
    context: RuntimeContext,
    *,
    options: Mapping[str, object],
    providers: RuntimeProviders | None = None,
) -> HisiliconModelSession:
    del context, options
    return HisiliconModelSession()


def _create_policy_session(
    context: RuntimeContext,
    options: Mapping[str, object],
    model_session_factory=None,
    *,
    session_registry=None,
    backend_registry=None,
    providers: RuntimeProviders | None = None,
):
    override = None
    if model_session_factory is not None:

        def override(builder_context, **_kwargs):
            return model_session_factory(builder_context, options)

    if session_registry is None:
        raise RuntimeDependencyError(
            "policy session construction requires an explicit session builder registry",
            code="session_builder_registry_required",
        )
    return session_registry.create(
        context,
        backend_registry=backend_registry,
        providers=providers,
        override=override,
        options=options,
    )


class _PolicyRuntimeAssembly(RuntimeAssembly):
    """Private bridge for the policy facade's remaining stage-based executor."""

    def __init__(self, policy_handle: _PolicySessionHandle, contract: str) -> None:
        self.policy_handle = policy_handle
        super().__init__(
            runtime_executor=policy_handle.model_executor,
            session=policy_handle._capability_source,
            execution_contract=contract,
            declared_capabilities={"stateful": False, "execution_contract": contract},
        )


def _policy_contract_name(context: RuntimeContext) -> str:
    declared = getattr(context.deployment, "execution_contract", None)
    name = getattr(declared, "name", None)
    if name is None and isinstance(declared, Mapping):
        name = declared.get("name") or declared.get("contract_name")
        if name is None and {"state_scope", "execution_structure"}.issubset(declared):
            name = f"{declared['state_scope']}-{declared['execution_structure']}"
    if isinstance(name, str) and name.strip():
        return name.strip()
    if context.model_type == "act":
        return "request-direct"
    return "request-iterative"


def _policy_visibility(context: RuntimeContext, contract: str) -> str | None:
    if contract.endswith("-direct"):
        return None
    declared = getattr(context.deployment, "execution_contract", None)
    value = getattr(declared, "orchestration_visibility", None)
    if value in {"executor", "session"}:
        return value
    return "session" if context.model_type == "diffusion" else "executor"


def _policy_runtime_key(context: RuntimeContext) -> ModelRuntimeKey:
    contract = _policy_contract_name(context)
    return ModelRuntimeKey(
        context.interface,
        context.model_type,
        context.operation,
        context.backend,
        contract,
        _policy_visibility(context, contract),
    )


def _policy_runtime_assembler(builder, contract: str):
    """Adapt one legacy policy stage builder to the public assembly boundary."""

    def assemble(
        context,
        model_session_factory=None,
        diagnostic_schedule=None,
        diagnostic_schedule_source=None,
        *,
        session_registry,
        backend_registry,
        providers,
    ) -> RuntimeAssembly:
        handle = builder(
            context,
            model_session_factory=model_session_factory,
            diagnostic_schedule=diagnostic_schedule,
            diagnostic_schedule_source=diagnostic_schedule_source,
            session_registry=session_registry,
            backend_registry=backend_registry,
            providers=providers,
        )
        if not isinstance(handle, _PolicySessionHandle):
            raise BackendLoadError(
                f"policy runtime assembler returned {type(handle).__name__}, expected a private policy handle",
                code="invalid_policy_assembly",
            )
        return _PolicyRuntimeAssembly(handle, contract)

    assemble.execution_contract = contract
    return assemble


def _resolve_pipeline_dependencies(
    *,
    registry_set: RegistrySet | None,
    providers: RuntimeProviders | None,
    backend_registry: BackendRegistry | None,
) -> tuple[BackendRegistry | object, object, RuntimeAssemblerRegistry, RuntimeProviders]:
    """Require the complete construction dependency set at the composition boundary."""

    if registry_set is None:
        raise RuntimeDependencyError(
            "create_inference_pipeline requires an explicitly injected RegistrySet",
            code="registry_set_required",
        )
    if providers is None:
        raise RuntimeDependencyError(
            "create_inference_pipeline requires explicitly injected RuntimeProviders",
            code="runtime_providers_required",
        )
    if not isinstance(registry_set, RegistrySet):
        raise RuntimeDependencyError(
            "create_inference_pipeline registry_set must be a RegistrySet",
            code="registry_set_invalid",
        )
    if not isinstance(providers, RuntimeProviders):
        raise RuntimeDependencyError(
            "create_inference_pipeline providers must be a RuntimeProviders value",
            code="runtime_providers_invalid",
        )
    return (
        registry_set.backend_registry if backend_registry is None else backend_registry,
        registry_set.session_builder_registry,
        registry_set.runtime_assembler_registry,
        providers,
    )


def create_inference_pipeline(
    pipeline_id: str,
    validated_manifest: ValidatedManifest,
    *,
    request_timeout: float | None = None,
    default_task: str | None = None,
    execution_mode: str = "monolithic",
    runtime_options: Mapping[str, object] | None = None,
    priority_scheduling: bool = False,
    registry: BackendRegistry | None = None,
    registry_set: RegistrySet | None = None,
    providers: RuntimeProviders | None = None,
    model_session_factory=None,
    pi05_diagnostic_schedule: PI05DenoisingSchedule | None = None,
    pi05_diagnostic_schedule_source: str | None = None,
    stage_policy: str = "sequential",
    stage_scheduling: Mapping[str, object] | None = None,
    frame_base_priority: int = 0,
) -> InferencePipeline:
    """Create one pipeline exclusively from a validated manifest and registry.

    Compiled Ascend PI0.5 deployments are routed through ``AscendOmModelSession``
    and the shared ``IterativeStage`` instead of the legacy backend-owned loop.
    Compiled HMM PI0.5, HMM SmolVLA, and RKNN SmolVLA deployments are likewise
    routed through ``HMMModelSession``/``RKNNModelSession`` and the shared
    family executors so compiled backends no longer own family loops.
    ``model_session_factory`` optionally overrides session construction for tests.
    """

    selected_registry, session_registry, assembler_registry, selected_providers = _resolve_pipeline_dependencies(
        registry_set=registry_set,
        providers=providers,
        backend_registry=registry,
    )
    context = RuntimeContext(
        validated_manifest,
        runtime_options=runtime_options or {},
        priority_scheduling=priority_scheduling,
    )
    selected_registry.validate(context)
    preprocessor = None
    postprocessor = None
    codec = None
    if isinstance(context.deployment, CompiledDeployment) and context.deployment.execution:
        preprocessor, postprocessor = create_lerobot_processor_views()
        codec = create_policy_codec(context.policy)
    assembly = _build_runtime_assembly(
        context,
        model_session_factory,
        pi05_diagnostic_schedule,
        pi05_diagnostic_schedule_source,
        session_registry=session_registry,
        backend_registry=selected_registry,
        assembler_registry=assembler_registry,
        providers=selected_providers,
    )
    if assembly is not None:
        handle = getattr(assembly, "policy_handle", None)
        if not isinstance(handle, _PolicySessionHandle):
            raise BackendLoadError(
                "policy runtime assembly does not expose its policy facade configuration",
                code="invalid_policy_assembly",
            )
        preprocessor = handle.preprocessor or preprocessor
        postprocessor = handle.postprocessor or postprocessor
        return InferencePipeline(
            pipeline_id,
            context,
            runtime_assembly=assembly,
            session_handle=handle,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            codec=codec,
            request_timeout=request_timeout,
            default_task=default_task,
            execution_mode=execution_mode,
            stage_policy=stage_policy,
            stage_scheduling=stage_scheduling,
            frame_base_priority=frame_base_priority,
        )
    raise BackendLoadError(
        f"v3 identity {context.interface}/{context.model_type}/{context.operation} with backend "
        f"{context.backend!r} "
        "has no registered model session factory",
        code="model_session_factory_unavailable",
    )


def create_pipeline_manager(
    pipeline_id: str,
    validated_manifest: ValidatedManifest,
    *,
    request_timeout: float | None = None,
    default_task: str | None = None,
    execution_mode: str = "monolithic",
    runtime_options: Mapping[str, object] | None = None,
    priority_scheduling: bool = False,
    registry: BackendRegistry | None = None,
    registry_set: RegistrySet | None = None,
    providers: RuntimeProviders | None = None,
    model_session_factory=None,
    pi05_diagnostic_schedule: PI05DenoisingSchedule | None = None,
    pi05_diagnostic_schedule_source: str | None = None,
    stage_policy: str = "sequential",
    stage_scheduling: Mapping[str, object] | None = None,
    frame_base_priority: int = 0,
) -> InferencePipelineManager:
    pipeline = create_inference_pipeline(
        pipeline_id,
        validated_manifest,
        request_timeout=request_timeout,
        default_task=default_task,
        execution_mode=execution_mode,
        runtime_options=runtime_options,
        priority_scheduling=priority_scheduling,
        registry=registry,
        registry_set=registry_set,
        providers=providers,
        model_session_factory=model_session_factory,
        pi05_diagnostic_schedule=pi05_diagnostic_schedule,
        pi05_diagnostic_schedule_source=pi05_diagnostic_schedule_source,
        stage_policy=stage_policy,
        stage_scheduling=stage_scheduling,
        frame_base_priority=frame_base_priority,
    )
    manager = InferencePipelineManager((pipeline,), retry_pending_close=priority_scheduling)
    manager.start()
    return manager


def _build_runtime_assembly(
    context: RuntimeContext,
    model_session_factory=None,
    diagnostic_schedule: PI05DenoisingSchedule | None = None,
    diagnostic_schedule_source: str | None = None,
    *,
    session_registry=None,
    backend_registry=None,
    assembler_registry: RuntimeAssemblerRegistry | None = None,
    providers: RuntimeProviders | None = None,
):
    """Construct the session-driven executor handle for compiled family backends.

    Compiled Ascend/HMM PI0.5 and HMM/RKNN SmolVLA deployments route through the
    matching :class:`ModelSession` and the shared family executor instead of a
    backend-owned loop.  ``model_session_factory`` optionally overrides session
    construction for tests.
    """

    if assembler_registry is None:
        raise RuntimeDependencyError(
            "policy pipeline construction requires an explicit runtime assembler registry",
            code="runtime_assembler_registry_required",
        )
    key = _policy_runtime_key(context)
    try:
        assembly = assembler_registry.assemble(
            key,
            context,
            model_session_factory,
            diagnostic_schedule,
            diagnostic_schedule_source,
            session_registry=session_registry,
            backend_registry=backend_registry,
            providers=providers,
        )
    except Exception as exc:
        if isinstance(exc, BackendLoadError):
            raise
        raise BackendLoadError(
            f"unable to assemble policy runtime for {key!r}: {exc}",
            code=getattr(exc, "code", "runtime_assembler_unavailable"),
        ) from exc
    if not isinstance(assembly, RuntimeAssembly):
        raise BackendLoadError(
            f"runtime assembler for {key!r} returned {type(assembly).__name__}, expected RuntimeAssembly",
            code="invalid_runtime_assembly",
        )
    if not isinstance(getattr(assembly, "policy_handle", None), _PolicySessionHandle):
        raise BackendLoadError(
            f"runtime assembler for {key!r} returned no private policy stage bridge",
            code="invalid_policy_assembly",
        )
    return assembly


def _build_torch_policy_handle(
    context: RuntimeContext,
    model_session_factory=None,
    diagnostic_schedule=None,
    diagnostic_schedule_source=None,
    *,
    session_registry=None,
    backend_registry=None,
    providers=None,
):
    """Construct a native LeRobot policy through its Torch model session."""

    del diagnostic_schedule, diagnostic_schedule_source
    deployment = context.deployment
    if not isinstance(deployment, TorchDeployment):
        raise BackendLoadError("LeRobot Torch policy requires a Torch deployment", code="invalid_deployment")
    LeRobotTorchModelSession.validate_runtime_options(context.runtime_options)
    session = _create_policy_session(
        context,
        context.runtime_options,
        model_session_factory,
        session_registry=session_registry,
        backend_registry=backend_registry,
        providers=providers,
    )
    model_executor = SequentialModelExecutor(
        (ModelStage("policy", session),),
        _RawActionResultAdapter("action"),
        components=(session,),
    )
    return _PolicySessionHandle(
        model_executor,
        context,
        "action",
        None,
        None,
        session,
        None,
        None,
        preprocessor=LeRobotSessionPreprocessor(session),
        postprocessor=LeRobotSessionPostprocessor(session),
        metadata_provider=session.execution_metadata,
    )


def _build_ascend_pi05_handle(
    context: RuntimeContext,
    model_session_factory=None,
    diagnostic_schedule: PI05DenoisingSchedule | None = None,
    diagnostic_schedule_source: str | None = None,
    *,
    session_registry=None,
    backend_registry=None,
    providers=None,
):
    """Construct the session-driven PI0.5 executor handle for compiled Ascend."""

    deployment = _compiled_deployment(context, "Ascend PI0.5")
    options = _validate_pi05_options(context.runtime_options)
    _validate_pi05_policy_config(context)
    plan = build_execution_plan(deployment.execution, deployment.bindings, deployment.device_links)
    action_binding = _session_action_binding(plan)
    schedule, schedule_source, schedule_override = _resolve_denoising_schedule(
        context,
        action_binding,
        diagnostic_schedule,
        diagnostic_schedule_source,
        require_velocity=options["curvature_log_path"] is not None,
    )
    _validate_schedule_compatibility(schedule, options["curvature_log_path"])
    num_inference_steps = _pi05_num_inference_steps(context)
    session = _create_ascend_session(
        context,
        options,
        model_session_factory,
        session_registry=session_registry,
        backend_registry=backend_registry,
        providers=providers,
    )
    velocity_trace: list[np.ndarray] | None = [] if options["curvature_log_path"] is not None else None
    model_executor = create_pi05_executor(
        plan,
        session,
        schedule,
        _RawActionResultAdapter(action_binding.semantic),
        num_inference_steps=num_inference_steps if schedule is None else None,
        initializer=_make_noise_initializer(plan, options["random_seed"]),
        velocity_trace=velocity_trace,
    )
    schedule_metadata = _schedule_metadata(schedule, schedule_source, schedule_override)
    session_context = _build_session_context(context, options)
    return _PolicySessionHandle(
        model_executor,
        session_context,
        action_binding.semantic,
        schedule_metadata,
        schedule,
        session,
        options["curvature_log_path"],
        velocity_trace,
    )


def _build_ascend_act_handle(
    context: RuntimeContext,
    model_session_factory=None,
    diagnostic_schedule=None,
    diagnostic_schedule_source=None,
    *,
    session_registry=None,
    backend_registry=None,
    providers=None,
):
    """Construct the single-role ACT executor through ``AscendOmModelSession``."""

    del diagnostic_schedule, diagnostic_schedule_source
    deployment = _compiled_deployment(context, "Ascend ACT")
    options = _validate_ascend_options(context.runtime_options)
    if deployment.execution != ("policy",):
        raise BackendLoadError(
            f"Ascend ACT requires execution ['policy'], got {list(deployment.execution)}",
            code="invalid_execution_plan",
        )
    plan = build_execution_plan(deployment.execution, deployment.bindings, deployment.device_links)
    action_binding = _session_action_binding(plan)
    session = _create_ascend_session(
        context,
        options,
        model_session_factory,
        session_registry=session_registry,
        backend_registry=backend_registry,
        providers=providers,
    )
    model_executor = SequentialModelExecutor(
        (ModelStage("policy", session),),
        _RawActionResultAdapter(action_binding.semantic),
        components=(session,),
        execution_plan=plan,
    )
    return _PolicySessionHandle(
        model_executor,
        _build_session_context(context, options),
        action_binding.semantic,
        None,
        None,
        session,
        None,
        None,
    )


def _build_hisilicon_act_handle(
    context: RuntimeContext,
    model_session_factory=None,
    diagnostic_schedule=None,
    diagnostic_schedule_source=None,
    *,
    session_registry=None,
    backend_registry=None,
    providers=None,
):
    """Construct the single-role ACT executor through ``HisiliconModelSession``."""

    del diagnostic_schedule, diagnostic_schedule_source
    deployment = _compiled_deployment(context, "Hisilicon ACT")
    from inference_service.model_sessions.hisilicon import validate_runtime_options

    options = validate_runtime_options(context.runtime_options)
    if deployment.execution != ("policy",):
        raise BackendLoadError(
            f"Hisilicon ACT requires execution ['policy'], got {list(deployment.execution)}",
            code="invalid_execution_plan",
        )
    plan = build_execution_plan(deployment.execution, deployment.bindings, deployment.device_links)
    action_binding = _session_action_binding(plan)
    session = _create_policy_session(
        context,
        options,
        model_session_factory,
        session_registry=session_registry,
        backend_registry=backend_registry,
        providers=providers,
    )
    model_executor = SequentialModelExecutor(
        (ModelStage("policy", session),),
        _RawActionResultAdapter(action_binding.semantic),
        components=(session,),
        execution_plan=plan,
    )
    return _PolicySessionHandle(
        model_executor,
        RuntimeContext(
            context.validated_manifest,
            runtime_options=options,
            priority_scheduling=context.priority_scheduling,
        ),
        action_binding.semantic,
        None,
        None,
        session,
        None,
        None,
        metadata_provider=session.execution_metadata,
    )


def _build_hmm_pi05_handle(
    context: RuntimeContext,
    model_session_factory=None,
    diagnostic_schedule=None,
    diagnostic_schedule_source=None,
    *,
    session_registry=None,
    backend_registry=None,
    providers=None,
):
    """Construct the session-driven PI0.5 executor handle for compiled HMM.

    The modular HMM PI0.5 topology declares a synthetic ``embedding`` host role
    and a ``time_mlp`` compiled module.  Embedding construction and sinusoidal
    timestep embedding run as executor-owned host stages while TCIM modules
    execute through ``HMMModelSession`` and the shared ``IterativeStage``.
    """

    del diagnostic_schedule, diagnostic_schedule_source
    deployment = _compiled_deployment(context, "HMM PI0.5")
    options = _validate_hmm_options(context.runtime_options)
    policy_config = load_pi05_policy_config(context)
    plan = build_execution_plan(deployment.execution, deployment.bindings, deployment.device_links)
    action_binding = _session_action_binding(plan)
    num_inference_steps = _hmm_pi05_num_inference_steps(policy_config)
    schedule = uniform_pi05_schedule(num_inference_steps)
    session = _create_hmm_session(
        context,
        options,
        model_session_factory,
        session_registry=session_registry,
        backend_registry=backend_registry,
        providers=providers,
    )
    family_resource = PI05HMMFamilyResource(deployment, policy_config)
    embedding_stage = build_embedding_stage(deployment, family_resource, policy_config)
    time_prep_stage = build_time_prep_stage(deployment, policy_config)
    model_executor = create_pi05_executor(
        plan,
        session,
        schedule,
        _RawActionResultAdapter(action_binding.semantic),
        initializer=_make_noise_initializer(plan, options["random_seed"]),
        embedding_stage=embedding_stage,
        time_prep_stage=time_prep_stage,
        extra_components=(family_resource,),
    )
    session_context = _build_hmm_session_context(context, options)
    return _PolicySessionHandle(
        model_executor,
        session_context,
        action_binding.semantic,
        _schedule_metadata(schedule, "uniform", False),
        schedule,
        session,
        None,
        None,
    )


def _validate_pi05_options(options: Mapping[str, object]) -> dict[str, object]:
    unknown = sorted(set(options) - _ALLOWED_PI05_OPTIONS)
    if unknown:
        raise BackendLoadError(f"unknown Ascend PI0.5 options: {unknown}", code="invalid_runtime_options")
    device_id = options.get("device_id", 0)
    if type(device_id) is not int or device_id < 0:
        raise BackendLoadError("Ascend device_id must be a non-negative integer", code="invalid_runtime_options")
    random_seed = options.get("random_seed")
    if random_seed is not None and type(random_seed) is not int:
        raise BackendLoadError("Ascend random_seed must be an integer or null", code="invalid_runtime_options")
    curvature_log_path = options.get("curvature_log_path")
    if curvature_log_path is not None and (type(curvature_log_path) is not str or not curvature_log_path.strip()):
        raise BackendLoadError("Ascend curvature_log_path must be a non-empty string", code="invalid_runtime_options")
    if curvature_log_path is not None:
        from pathlib import Path

        path = Path(str(curvature_log_path)).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.open("a", encoding="utf-8").close()
    return {
        "device_id": device_id,
        "random_seed": random_seed,
        "curvature_log_path": curvature_log_path,
    }


def _validate_ascend_options(options: Mapping[str, object]) -> dict[str, object]:
    unknown = sorted(set(options) - _ALLOWED_ASCEND_OPTIONS)
    if unknown:
        raise BackendLoadError(f"unknown Ascend options: {unknown}", code="invalid_runtime_options")
    device_id = options.get("device_id", 0)
    if type(device_id) is not int or device_id < 0:
        raise BackendLoadError("Ascend device_id must be a non-negative integer", code="invalid_runtime_options")
    return {"device_id": device_id}


def _session_action_binding(plan) -> TensorBinding:
    action_role = next(role for role in plan.roles if any(b.semantic == "action" for b in role.bindings.outputs))
    matches = [b for b in action_role.bindings.outputs if b.semantic == "action"]
    if len(matches) != 1:
        raise BackendLoadError(
            "compiled deployment requires exactly one action output binding", code="invalid_action_binding"
        )
    return matches[0]


def _resolve_denoising_schedule(
    context: RuntimeContext,
    action_binding: TensorBinding,
    diagnostic_schedule: PI05DenoisingSchedule | None,
    diagnostic_schedule_source: str | None,
    require_velocity: bool = False,
):
    deployment = _compiled_deployment(context, "Ascend PI0.5 schedule")
    runtime_name = action_binding.runtime_name or ""
    output_name = next(
        (part for part in reversed(runtime_name.split(":")) if part in {"action", "velocity", "v_t"}),
        runtime_name,
    )
    velocity_mode = output_name in {"velocity", "v_t"}
    if (diagnostic_schedule is not None or require_velocity) and not velocity_mode:
        raise BackendLoadError(
            "Ascend PI0.5 schedule diagnostics require a velocity/v_t Action Expert runtime output",
            code="invalid_runtime_options",
        )
    if diagnostic_schedule is not None:
        return diagnostic_schedule, diagnostic_schedule_source or "diagnostic", True
    artifact = deployment.artifacts.get("denoising_schedule")
    if artifact is None:
        if velocity_mode:
            raise BackendLoadError(
                "Ascend PI0.5 velocity output requires a denoising_schedule artifact",
                code="missing_denoising_schedule",
            )
        return None, None, False
    if artifact.format != "json":
        raise BackendLoadError(
            "Ascend PI0.5 denoising_schedule artifact must use format 'json'", code="invalid_denoising_schedule"
        )
    if not velocity_mode:
        raise BackendLoadError(
            "Ascend PI0.5 denoising_schedule requires a velocity/v_t Action Expert runtime output",
            code="invalid_denoising_schedule",
        )
    try:
        schedule = load_pi05_schedule(context.resolved_artifacts["denoising_schedule"])
    except Exception as exc:
        raise BackendLoadError(
            f"Unable to load PI0.5 denoising schedule {artifact.path}: {exc}", code="invalid_denoising_schedule"
        ) from exc
    return schedule, artifact.path, False


def _validate_pi05_policy_config(context: RuntimeContext) -> None:
    config = _load_policy_config(context)
    for key in ("chunk_size", "max_action_dim", "num_inference_steps"):
        value = config.get(key)
        if type(value) is not int or value < 1:
            raise BackendLoadError(
                f"Ascend PI0.5 requires positive integer {key!r} in LeRobot config",
                code="invalid_policy_config",
            )


def _validate_schedule_compatibility(schedule: PI05DenoisingSchedule | None, curvature_log_path) -> None:
    if curvature_log_path is not None and schedule is None:
        raise BackendLoadError(
            "Ascend PI0.5 curvature log requires a velocity/v_t Action Expert runtime output and schedule",
            code="invalid_runtime_options",
        )


def _pi05_num_inference_steps(context: RuntimeContext) -> int:
    config = _load_policy_config(context)
    value = config.get("num_inference_steps")
    if type(value) is not int or value < 1:
        raise BackendLoadError(
            "Ascend PI0.5 requires positive integer 'num_inference_steps' in LeRobot config",
            code="invalid_policy_config",
        )
    return value


def _load_policy_config(context: RuntimeContext) -> dict[str, object]:
    try:
        value = load_json_strict(context.validated_manifest.bundle_root / "config.json")
    except Exception as exc:
        raise BackendLoadError(f"Unable to read LeRobot config: {exc}", code="invalid_policy_config") from exc
    if not isinstance(value, dict):
        raise BackendLoadError("LeRobot config must be an object", code="invalid_policy_config")
    return value


def _create_ascend_session(
    context: RuntimeContext,
    options: Mapping[str, object],
    model_session_factory=None,
    *,
    session_registry=None,
    backend_registry=None,
    providers=None,
) -> AscendOmModelSession:
    return _create_policy_session(
        context,
        options,
        model_session_factory,
        session_registry=session_registry,
        backend_registry=backend_registry,
        providers=providers,
    )


def _build_session_context(context: RuntimeContext, options: Mapping[str, object]) -> RuntimeContext:
    session_options: dict[str, object] = {"device_id": options["device_id"]}
    return RuntimeContext(
        context.validated_manifest,
        runtime_options=session_options,
        priority_scheduling=context.priority_scheduling,
        runtime_profile=context.runtime_profile,
    )


def _validate_hmm_options(options: Mapping[str, object]) -> dict[str, object]:
    unknown = sorted(set(options) - _ALLOWED_HMM_OPTIONS)
    if unknown:
        raise BackendLoadError(f"unknown HMM options: {unknown}", code="invalid_runtime_options")
    device_id = options.get("device_id", 0)
    if type(device_id) is not int or device_id < 0:
        raise BackendLoadError("HMM device_id must be a non-negative integer", code="invalid_runtime_options")
    random_seed = options.get("random_seed")
    if random_seed is not None and type(random_seed) is not int:
        raise BackendLoadError("HMM random_seed must be an integer or null", code="invalid_runtime_options")
    return {"device_id": device_id, "random_seed": random_seed}


def _hmm_pi05_num_inference_steps(policy_config: Mapping[str, object]) -> int:
    value = policy_config.get("num_inference_steps")
    if type(value) is not int or value < 1:
        raise BackendLoadError(
            "HMM PI0.5 requires positive integer 'num_inference_steps' in LeRobot config",
            code="invalid_policy_config",
        )
    return value


def _create_hmm_session(
    context: RuntimeContext,
    options: Mapping[str, object],
    model_session_factory=None,
    *,
    session_registry=None,
    backend_registry=None,
    providers=None,
) -> HMMModelSession:
    return _create_policy_session(
        context,
        options,
        model_session_factory,
        session_registry=session_registry,
        backend_registry=backend_registry,
        providers=providers,
    )


def _build_hmm_session_context(context: RuntimeContext, options: Mapping[str, object]) -> RuntimeContext:
    session_options: dict[str, object] = {"device_id": options["device_id"]}
    return RuntimeContext(
        context.validated_manifest,
        runtime_options=session_options,
        priority_scheduling=context.priority_scheduling,
    )


def _make_noise_initializer(plan, random_seed):
    topology = derive_pi05_topology(plan)
    noise_binding = next(
        b
        for role in topology.loop_roles
        for b in plan.role(role).bindings.inputs
        if b.semantic == topology.state_semantic
    )
    rng = np.random.default_rng(random_seed)

    def initializer(values: Mapping[str, object]) -> np.ndarray:
        return np.ascontiguousarray(rng.standard_normal(noise_binding.shape).astype(np.dtype(noise_binding.dtype)))

    return initializer


def _build_hmm_smolvla_handle(
    context: RuntimeContext,
    model_session_factory=None,
    diagnostic_schedule=None,
    diagnostic_schedule_source=None,
    *,
    session_registry=None,
    backend_registry=None,
    providers=None,
):
    """Construct the session-driven SmolVLA executor handle for compiled HMM.

    SmolVLA HMM declares device-resident KV links between ``prefill`` and
    ``action``.  The family executor (``create_smolvla_executor``) runs vision,
    a synthetic host ``embedding`` role, ``prefill``, and the iterative
    ``action`` body through :class:`HMMModelSession`, which keeps the declared
    device links device-resident.  Embedding weights load once into the
    executor-owned :class:`SmolVLAFamilyResource`.
    """

    del diagnostic_schedule, diagnostic_schedule_source
    deployment = _compiled_deployment(context, "HMM SmolVLA")
    options = _validate_hmm_options(context.runtime_options)
    policy_config = load_smolvla_policy_config(context)
    plan = build_execution_plan(deployment.execution, deployment.bindings, deployment.device_links)
    action_binding = _session_action_binding(plan)
    num_inference_steps = _smolvla_num_inference_steps(policy_config)
    session = _create_hmm_session(
        context,
        options,
        model_session_factory,
        session_registry=session_registry,
        backend_registry=backend_registry,
        providers=providers,
    )
    resource = SmolVLAFamilyResource(deployment, policy_config)
    model_executor = create_smolvla_executor(
        plan,
        session,
        resource,
        _RawActionResultAdapter(action_binding.semantic),
        num_inference_steps=num_inference_steps,
        initializer=_make_smolvla_noise_initializer(plan, options["random_seed"]),
    )
    return _PolicySessionHandle(
        model_executor,
        _build_hmm_session_context(context, options),
        action_binding.semantic,
        None,
        None,
        session,
        None,
        None,
    )


def _build_rknn_smolvla_handle(
    context: RuntimeContext,
    model_session_factory=None,
    diagnostic_schedule=None,
    diagnostic_schedule_source=None,
    *,
    session_registry=None,
    backend_registry=None,
    providers=None,
):
    """Construct the session-driven SmolVLA executor handle for compiled RKNN.

    SmolVLA RKNN declares no device links; prefix/KV tensors are host-visible
    bindings threaded through the executor ``ExecutionFrame``.  Runtime-specific
    tensor conversion stays below :class:`RKNNModelSession`, which exposes only
    semantic role invocation to the shared family executor.
    """

    del diagnostic_schedule, diagnostic_schedule_source
    deployment = _compiled_deployment(context, "RKNN SmolVLA")
    options = _validate_rknn_options(context.runtime_options)
    policy_config = load_smolvla_policy_config(context)
    plan = build_execution_plan(deployment.execution, deployment.bindings, deployment.device_links)
    action_binding = _session_action_binding(plan)
    num_inference_steps = _smolvla_num_inference_steps(policy_config)
    session = _create_rknn_session(
        context,
        options,
        model_session_factory,
        session_registry=session_registry,
        backend_registry=backend_registry,
        providers=providers,
    )
    resource = SmolVLAFamilyResource(deployment, policy_config)
    model_executor = create_smolvla_executor(
        plan,
        session,
        resource,
        _RawActionResultAdapter(action_binding.semantic),
        num_inference_steps=num_inference_steps,
        initializer=_make_smolvla_noise_initializer(plan, options["random_seed"]),
    )
    return _PolicySessionHandle(
        model_executor,
        _build_rknn_session_context(context, options),
        action_binding.semantic,
        None,
        None,
        session,
        None,
        None,
    )


def _build_rknn_act_handle(
    context: RuntimeContext,
    model_session_factory=None,
    diagnostic_schedule=None,
    diagnostic_schedule_source=None,
    *,
    session_registry=None,
    backend_registry=None,
    providers=None,
):
    """Construct the single-role ACT executor through ``RKNNModelSession``."""

    del diagnostic_schedule, diagnostic_schedule_source
    deployment = _compiled_deployment(context, "RKNN ACT")
    options = _validate_rknn_options(context.runtime_options)
    if deployment.execution != ("policy",):
        raise BackendLoadError(
            f"RKNN ACT requires execution ['policy'], got {list(deployment.execution)}",
            code="invalid_execution_plan",
        )
    plan = build_execution_plan(deployment.execution, deployment.bindings, deployment.device_links)
    action_binding = _session_action_binding(plan)
    session = _create_rknn_session(
        context,
        options,
        model_session_factory,
        session_registry=session_registry,
        backend_registry=backend_registry,
        providers=providers,
    )
    model_executor = SequentialModelExecutor(
        (ModelStage("policy", session),),
        _RawActionResultAdapter(action_binding.semantic),
        components=(session,),
        execution_plan=plan,
    )
    return _PolicySessionHandle(
        model_executor,
        _build_rknn_session_context(context, options),
        action_binding.semantic,
        None,
        None,
        session,
        None,
        None,
    )


def _smolvla_num_inference_steps(policy_config: Mapping[str, object]) -> int:
    value = policy_config.get("num_steps")
    if type(value) is not int or value < 1:
        raise BackendLoadError(
            "SmolVLA requires positive integer 'num_steps' in LeRobot config",
            code="invalid_policy_config",
        )
    return value


def _make_smolvla_noise_initializer(plan, random_seed):
    topology = derive_smolvla_topology(plan)
    noise_binding = next(
        binding for binding in plan.role("action").bindings.inputs if binding.semantic == topology.state_semantic
    )
    rng = np.random.default_rng(random_seed)

    def initializer(values: Mapping[str, object]) -> np.ndarray:
        return np.ascontiguousarray(rng.standard_normal(noise_binding.shape).astype(np.float32))

    return initializer


def _validate_rknn_options(options: Mapping[str, object]) -> dict[str, object]:
    unknown = sorted(set(options) - _ALLOWED_RKNN_OPTIONS)
    if unknown:
        raise BackendLoadError(f"unknown RKNN options: {unknown}", code="invalid_runtime_options")
    return validate_rknn_runtime_options(options)


def _create_rknn_session(
    context: RuntimeContext,
    options: Mapping[str, object],
    model_session_factory=None,
    *,
    session_registry=None,
    backend_registry=None,
    providers=None,
) -> RKNNModelSession:
    return _create_policy_session(
        context,
        options,
        model_session_factory,
        session_registry=session_registry,
        backend_registry=backend_registry,
        providers=providers,
    )


def _build_rknn_session_context(context: RuntimeContext, options: Mapping[str, object]) -> RuntimeContext:
    session_options: dict[str, object] = {"target": options["target"], "core_mask": options["core_mask"]}
    return RuntimeContext(
        context.validated_manifest,
        runtime_options=session_options,
        priority_scheduling=context.priority_scheduling,
    )


def _schedule_metadata(schedule, source, override):
    if schedule is None:
        return None
    metadata = {"name": schedule.name, "step_count": schedule.step_count, "source": source}
    if override:
        metadata["override"] = True
    return metadata


def _compiled_deployment(context: RuntimeContext, family: str) -> CompiledDeployment:
    deployment = context.deployment
    if not isinstance(deployment, CompiledDeployment):
        raise BackendLoadError(f"{family} requires a compiled deployment", code="invalid_deployment")
    return deployment


_POLICY_EXECUTOR_BUILDERS = (
    ("act", "torch", _build_torch_policy_handle),
    ("act", "ascend", _build_ascend_act_handle),
    ("act", "rknn", _build_rknn_act_handle),
    ("act", "hisilicon", _build_hisilicon_act_handle),
    ("diffusion", "torch", _build_torch_policy_handle),
    ("pi05", "torch", _build_torch_policy_handle),
    ("pi05", "ascend", _build_ascend_pi05_handle),
    ("pi05", "hmm", _build_hmm_pi05_handle),
    ("smolvla", "hmm", _build_hmm_smolvla_handle),
    ("smolvla", "rknn", _build_rknn_smolvla_handle),
    ("smolvla", "torch", _build_torch_policy_handle),
)

_POLICY_SESSION_BUILDERS = (
    ("act", "torch", _new_lerobot_torch_session),
    ("act", "ascend", _new_ascend_session),
    ("act", "rknn", _new_rknn_session),
    ("act", "hisilicon", _new_hisilicon_session),
    ("diffusion", "torch", _new_lerobot_torch_session),
    ("pi05", "torch", _new_lerobot_torch_session),
    ("pi05", "ascend", _new_ascend_session),
    ("pi05", "hmm", _new_hmm_session),
    ("smolvla", "hmm", _new_hmm_session),
    ("smolvla", "rknn", _new_rknn_session),
    ("smolvla", "torch", _new_lerobot_torch_session),
)


_POLICY_PROFILE_TYPES = {
    "torch": TorchRuntimeProfile,
    "ascend": AscendRuntimeProfile,
    "hmm": HMMRuntimeProfile,
    "rknn": RKNNRuntimeProfile,
    "hisilicon": HisiliconRuntimeProfile,
}
_POLICY_TARGET_RUNTIMES = {
    "torch": frozenset({"torch"}),
    "ascend": frozenset({"acl"}),
    "hmm": frozenset({"hmm", "tcim"}),
    "rknn": frozenset({"rknn", "rknn-lite", "rknn-lite2"}),
    "hisilicon": frozenset({"hisilicon-worker"}),
}
_POLICY_CONTRACTS = {
    "act": "request-direct",
    "diffusion": "request-iterative",
    # Torch LeRobot policies execute one complete prediction per request. The
    # iterative contract is reserved for compiled family runtimes.
    "pi05": "request-direct",
    "smolvla": "request-direct",
}


def register_policy_session_builders(
    session_registry=None,
    assembler_registry: RuntimeAssemblerRegistry | None = None,
) -> None:
    """Register policy Sessions and role assemblers into an explicit RegistrySet."""

    if session_registry is None:
        raise RuntimeDependencyError(
            "register_policy_session_builders requires a session registry",
            code="session_builder_registry_required",
        )
    if assembler_registry is None:
        raise RuntimeDependencyError(
            "register_policy_session_builders requires a runtime assembler registry",
            code="runtime_assembler_registry_required",
        )
    for model_type, backend, session_builder in _POLICY_SESSION_BUILDERS:
        if session_registry.get("policy", model_type, "predict", backend) is None:
            session_registry.register("policy", model_type, "predict", backend, session_builder)

    for model_type, backend, builder in _POLICY_EXECUTOR_BUILDERS:
        contract = _POLICY_CONTRACTS[model_type]
        visibility = "session" if contract.endswith("iterative") else None
        contracts = [(contract, visibility)]
        # Compiled family loops are driven by IterativeStage. Keep the existing
        # direct registration for deployments that already select that contract.
        if model_type in {"pi05", "smolvla"} and backend != "torch":
            contracts.append(("request-iterative", "executor"))
        for contract, visibility in contracts:
            key = ModelRuntimeKey("policy", model_type, "predict", backend, contract, visibility)
            if assembler_registry.get(key) is not None:
                continue
            assembler_registry.register(
                RuntimeDescriptor(
                    key=key,
                    session_builder_key=SessionBuilderKey("policy", model_type, "predict", backend),
                    profile_type=_POLICY_PROFILE_TYPES[backend],
                    assembler=_policy_runtime_assembler(builder, contract),
                    execution_contract=contract,
                    declared_capabilities={"stateful": False, "execution_contract": contract},
                    supported_target_runtimes=_POLICY_TARGET_RUNTIMES[backend],
                )
            )

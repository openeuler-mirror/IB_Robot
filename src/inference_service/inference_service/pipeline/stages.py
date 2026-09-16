"""ROS-independent stage contracts for generic model execution."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from inference_service.pipeline.runtime_core import StageFrame


@runtime_checkable
class InferenceStage(Protocol):
    """One ordered operation over a request-local stage frame."""

    def execute(self, frame: StageFrame, *, deadline: datetime | None) -> None: ...


@runtime_checkable
class ResultAdapter(Protocol):
    """Convert the final stage frame into a domain result."""

    def adapt(self, frame: StageFrame) -> object: ...

    def adapt_error(self, error: object) -> object: ...


@dataclass(frozen=True)
class ModelResultAdapter:
    """Return the named-tensor result produced by a :class:`ModelStage`."""

    result_key: str = "_model_result"

    def adapt(self, frame: StageFrame) -> object:
        try:
            return frame.values[self.result_key]
        except KeyError as exc:
            raise RuntimeError(f"stage frame is missing model result {self.result_key!r}") from exc

    def adapt_error(self, error: object) -> object:
        cause = getattr(error, "cause", None)
        if cause is not None:
            raise cause
        raise RuntimeError(getattr(error, "message", str(error)))


@runtime_checkable
class IterationStateAdapter(Protocol):
    """Own family-specific loop state without coupling it to a backend."""

    def initialize(self, frame: StageFrame) -> None: ...

    def prepare_step(self, frame: StageFrame, step: IterationStep) -> None: ...

    def update(self, frame: StageFrame, step: IterationStep) -> None: ...

    def finalize(self, frame: StageFrame) -> None: ...


@dataclass(frozen=True)
class IterationStep:
    """One iterative-model step and its state-update delta."""

    index: int
    timestep: float
    delta: float

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("iteration index cannot be negative")


@dataclass(frozen=True)
class IterativeStage:
    """Repeat a deployment-derived stage body using a family state adapter."""

    plan: Iterable[IterationStep]
    body: tuple[InferenceStage, ...]
    state_adapter: IterationStateAdapter
    loop_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan", tuple(self.plan))
        if not self.plan:
            raise ValueError("iterative stage plan cannot be empty")
        if not self.body:
            raise ValueError("iterative stage body cannot be empty")

    def execute(self, frame: StageFrame, *, deadline: datetime | None) -> None:
        frame.control.raise_if_canceled("iterative.initialize")
        if frame.execution_frame is not None:
            if not self.loop_roles:
                raise ValueError("iterative stage with an execution plan requires loop_roles")
            frame.execution_frame.configure_loop(self.loop_roles, len(self.plan))
        self.state_adapter.initialize(frame)
        for step in self.plan:
            frame.control.raise_if_canceled(f"iterative.step.{step.index}")
            self.state_adapter.prepare_step(frame, step)
            for stage in self.body:
                stage.execute(frame, deadline=deadline)
            self.state_adapter.update(frame, step)
        self.state_adapter.finalize(frame)


@dataclass(frozen=True)
class EulerIterationStateAdapter:
    """Apply float32 Euler updates to semantic loop state."""

    state_semantic: str = "noise"
    velocity_semantic: str = "velocity"
    timestep_semantic: str = "timestep"
    result_semantic: str = "action"
    initializer: Callable[[Mapping[str, object]], np.ndarray] | None = None
    timestep_shape: tuple[int, ...] | None = None
    velocity_trace: list[np.ndarray] | None = None

    def initialize(self, frame: StageFrame) -> None:
        if self.velocity_trace is not None:
            self.velocity_trace.clear()
        value = frame.values.get(self.state_semantic)
        if value is None and self.initializer is not None:
            value = self.initializer(frame.values)
        if not isinstance(value, np.ndarray):
            raise ValueError(f"iterative state {self.state_semantic!r} must be a NumPy array")
        frame.values[self.state_semantic] = np.ascontiguousarray(value, dtype=np.float32)

    def prepare_step(self, frame: StageFrame, step: IterationStep) -> None:
        frame.values[self.timestep_semantic] = _timestep_tensor(step.timestep, self.timestep_shape)

    def update(self, frame: StageFrame, step: IterationStep) -> None:
        state = frame.values.get(self.state_semantic)
        velocity = frame.values.get(self.velocity_semantic)
        if not isinstance(state, np.ndarray) or not isinstance(velocity, np.ndarray):
            raise ValueError("Euler update requires NumPy state and velocity values")
        if state.shape != velocity.shape:
            raise ValueError(f"Euler state and velocity shapes differ: {state.shape} != {velocity.shape}")
        if self.velocity_trace is not None:
            self.velocity_trace.append(np.asarray(velocity, dtype=np.float32))
        frame.values[self.state_semantic] = np.ascontiguousarray(
            state.astype(np.float32, copy=False) + np.float32(step.delta) * velocity.astype(np.float32, copy=False)
        )

    def finalize(self, frame: StageFrame) -> None:
        frame.values[self.result_semantic] = frame.values[self.state_semantic]


@dataclass(frozen=True)
class DirectIterationStateAdapter:
    """Replace iterative state with each model output."""

    state_semantic: str = "noise"
    output_semantic: str = "action"
    timestep_semantic: str = "timestep"
    result_semantic: str = "action"
    initializer: Callable[[Mapping[str, object]], np.ndarray] | None = None
    timestep_shape: tuple[int, ...] | None = None

    def initialize(self, frame: StageFrame) -> None:
        value = frame.values.get(self.state_semantic)
        if value is None and self.initializer is not None:
            value = self.initializer(frame.values)
        if not isinstance(value, np.ndarray):
            raise ValueError(f"iterative state {self.state_semantic!r} must be a NumPy array")
        frame.values[self.state_semantic] = np.ascontiguousarray(value)

    def prepare_step(self, frame: StageFrame, step: IterationStep) -> None:
        frame.values[self.timestep_semantic] = _timestep_tensor(step.timestep, self.timestep_shape)

    def update(self, frame: StageFrame, step: IterationStep) -> None:
        del step
        value = frame.values.get(self.output_semantic)
        if not isinstance(value, np.ndarray):
            raise ValueError(f"iterative output {self.output_semantic!r} must be a NumPy array")
        frame.values[self.state_semantic] = np.ascontiguousarray(value)

    def finalize(self, frame: StageFrame) -> None:
        frame.values[self.result_semantic] = frame.values[self.state_semantic]


@dataclass(frozen=True)
class PreprocessStage:
    operation: Callable[[Mapping[str, object]], Mapping[str, object]]

    def execute(self, frame: StageFrame, *, deadline: datetime | None) -> None:
        del deadline
        frame.control.raise_if_canceled("preprocess")
        frame.values.update(self.operation(frame.values))


@dataclass(frozen=True)
class ModelStage:
    role: str
    session: object

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("model stage role must be non-empty")

    def execute(self, frame: StageFrame, *, deadline: datetime | None) -> None:
        model_request = frame.values.get("_model_request")
        request = model_request or frame.request
        frame.control.raise_if_canceled(f"model.{self.role}")
        from inference_service.unified_runtime import ExecutionContext, ModelRequest

        if not isinstance(request, ModelRequest):
            raise TypeError("ModelStage requires a ModelRequest")
        execution_context = frame.values.get("_execution_context")
        if not isinstance(execution_context, ExecutionContext):
            raise TypeError("ModelStage requires the native ExecutionContext")
        if frame.execution_frame is None:
            inputs = request.inputs if model_request is not None else frame.values
            result = self.session.execute(ModelRequest(inputs, request.metadata), execution_context)
            outputs = result.outputs if hasattr(result, "outputs") else result
            frame.values["_model_result"] = result
            frame.values.update(outputs)
            return
        role_inputs = frame.execution_frame.begin_role(self.role)
        values = {**role_inputs, **frame.values}
        plan = frame.execution_plan
        linked_inputs = (
            set()
            if frame.values.get("_independent_stage")
            else {link.semantic for link in plan.device_links_for_consumer(self.role)}
        )
        role_input_bindings = plan.role(self.role).bindings.inputs
        host_semantics = {binding.semantic for binding in role_input_bindings if binding.semantic not in linked_inputs}
        binding_by_semantic = {binding.semantic: binding for binding in role_input_bindings}
        selected_values = {
            semantic: _coerce_input_dtype(values[semantic], binding_by_semantic[semantic])
            for semantic in host_semantics
            if semantic in values
        }
        options = {}
        if frame.values.get("_independent_stage"):
            priority_offsets = frame.values.get("_stage_priority_offsets", {})
            priority = frame.values.get("_stage_base_priority", request.metadata.get("priority", 0))
            if isinstance(priority_offsets, Mapping):
                priority += int(priority_offsets.get(self.role, 0))
            request = ModelRequest(
                selected_values,
                {**request.metadata, "priority": priority},
            )
            options = {"isolated": True, "operation_factory": frame.values.get("_operation_factory")}
        execute_role = getattr(self.session, "execute_role", None)
        if not callable(execute_role):
            raise TypeError(f"session {type(self.session).__name__} does not support role execution")
        outputs = execute_role(
            self.role,
            selected_values,
            ModelRequest(selected_values, request.metadata),
            execution_context,
            **options,
        )
        frame.execution_frame.finish_role(self.role, outputs)
        frame.values.update(
            {semantic: value for semantic, value in outputs.items() if not semantic.startswith("internal.")}
        )


@dataclass(frozen=True)
class HostComputeStage:
    operation: Callable[[Mapping[str, object]], Mapping[str, object]]

    def execute(self, frame: StageFrame, *, deadline: datetime | None) -> None:
        del deadline
        frame.control.raise_if_canceled("host_compute")
        frame.values.update(self.operation(frame.values))


@dataclass(frozen=True)
class HostRoleStage:
    """Execute a manifest-declared host role through deterministic host computation.

    Unlike :class:`HostComputeStage`, this stage participates in the
    :class:`ExecutionFrame` ``begin_role``/``finish_role`` traversal so that
    host-visible internal outputs declared by a synthetic manifest role are
    published, validated, and released in the same order as device-role
    outputs.  The ``operation`` callable receives the merged host-linked and
    frame values and returns the role's declared semantic outputs.
    """

    role: str
    operation: Callable[[Mapping[str, object]], Mapping[str, object]]

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("host role stage role must be non-empty")

    def execute(self, frame: StageFrame, *, deadline: datetime | None) -> None:
        del deadline
        frame.control.raise_if_canceled(f"host_role.{self.role}")
        if frame.execution_frame is None:
            frame.values.update(self.operation(frame.values))
            return
        role_inputs = frame.execution_frame.begin_role(self.role)
        values = {**role_inputs, **frame.values}
        outputs = self.operation(values)
        frame.execution_frame.finish_role(self.role, outputs)
        frame.values.update(
            {semantic: value for semantic, value in outputs.items() if not semantic.startswith("internal.")}
        )


@dataclass(frozen=True)
class PostprocessStage:
    operation: Callable[[Mapping[str, object]], Mapping[str, object]]

    def execute(self, frame: StageFrame, *, deadline: datetime | None) -> None:
        del deadline
        frame.control.raise_if_canceled("postprocess")
        frame.values.update(self.operation(frame.values))


def _timestep_tensor(timestep: float, shape: tuple[int, ...] | None) -> np.ndarray:
    if not shape:
        return np.float32(timestep)
    return np.ascontiguousarray(np.full(shape, timestep, dtype=np.float32))


def _coerce_input_dtype(value: object, binding) -> object:
    """Convert a host-stage value to the declared binding dtype when safe.

    Iterative state adapters (e.g. Euler) operate in float32 for numerical
    stability, but compiled module bindings may declare float16.  The coercion
    happens at the ModelStage boundary so the session receives values matching
    its manifest-declared ABI.
    """

    if not isinstance(value, np.ndarray):
        return value
    try:
        target = np.dtype(binding.dtype)
    except TypeError:
        return value
    if value.dtype == target:
        return value
    try:
        return np.ascontiguousarray(value, dtype=target)
    except (TypeError, ValueError):
        return value

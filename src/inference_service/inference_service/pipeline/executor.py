"""Concrete ordered executor built from manifest-aware runtime stages."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime

from inference_service.backends.types import BackendHealth, BackendState, RuntimeContext
from inference_service.pipeline.runtime_core import ExecutionControl, ExecutionError, ModelExecutor, StageFrame
from inference_service.pipeline.stages import InferenceStage, ResultAdapter
from inference_service.unified_runtime import ExecutionContext, ModelRequest


class ComponentModelExecutor(ModelExecutor):
    """Shared component ownership views and stage execution helpers."""

    supports_priority_zero_deadline_admission = False

    def __init__(
        self,
        stages: Iterable[InferenceStage],
        result_adapter: ResultAdapter,
        *,
        components: Iterable[object] = (),
        execution_plan=None,
        component_contexts: Mapping[int, RuntimeContext] | None = None,
        error_handler: Callable[[Exception, bool], None] | None = None,
        health_override: Callable[[], BackendHealth | None] | None = None,
        execution_contract: str | None = None,
        orchestration_visibility: str | None = None,
    ) -> None:
        self._stages = tuple(stages)
        if not self._stages:
            raise ValueError("executor requires at least one stage")
        self._result_adapter = result_adapter
        self._components = self._unique(tuple(components))
        self._execution_plan = execution_plan
        self._component_contexts = dict(component_contexts or {})
        self._error_handler = error_handler
        self._health_override = health_override
        self._execution_contract = execution_contract
        self._orchestration_visibility = orchestration_visibility
        self._context: RuntimeContext | None = None

    @property
    def stages(self) -> tuple[InferenceStage, ...]:
        return self._stages

    @property
    def components(self) -> tuple[object, ...]:
        return self._components

    @property
    def execution_plan(self):
        return self._execution_plan

    @property
    def execution_contract(self) -> str | None:
        return self._execution_contract

    @property
    def orchestration_visibility(self) -> str | None:
        return self._orchestration_visibility

    @property
    def component_contexts(self) -> Mapping[int, RuntimeContext]:
        return self._component_contexts

    def load(self, context: RuntimeContext) -> None:
        del context
        # Components are owned and loaded by ModelRuntimeHandle.

    def _execute_stages(self, request: ModelRequest, context: ExecutionContext) -> object:
        if not isinstance(request, ModelRequest):
            raise TypeError("SequentialModelExecutor requires a ModelRequest")
        if not isinstance(context, ExecutionContext):
            raise TypeError("SequentialModelExecutor requires an ExecutionContext")
        control = ExecutionControl(context.request_id, context.cancellation_token)
        frame = StageFrame(
            request,
            execution_plan=self._execution_plan,
            values={**request.inputs, "_execution_context": context},
            control=control,
        )
        try:
            for stage in self._stages:
                context.check("stage")
                stage.execute(frame, deadline=context.deadline.expires_at)
            return self._result_adapter.adapt(frame)
        except Exception as exc:
            if self._error_handler is not None:
                self._error_handler(exc, bool(frame.values.get("_backend_started", False)))
            raise
        finally:
            frame.close()

    def cancel(self, request_id: str, deadline: datetime | None = None) -> None:
        supported = False
        for component in self._components:
            cancel = getattr(component, "cancel", None)
            capabilities = getattr(component, "capabilities", None)
            if not callable(cancel) or not getattr(capabilities, "supports_cancellation", False):
                continue
            supported = True
            cancel(request_id, deadline=deadline)
        if not supported:
            # The request control is still honored by execute(); cancellation
            # is best-effort when the underlying session cannot interrupt work.
            return

    def adapt_error(self, error: ExecutionError) -> object:
        return self._result_adapter.adapt_error(error)

    def health(self) -> BackendHealth:
        if self._health_override is not None:
            override = self._health_override()
            if override is not None:
                return override
        healths = [component.health() for component in self._components if callable(getattr(component, "health", None))]
        if not healths:
            return BackendHealth(state=BackendState.READY, ready=True)
        failed = next((health for health in healths if not health.ready), None)
        if failed is not None:
            return BackendHealth(
                state=failed.state,
                ready=False,
                reason_code=failed.reason_code,
                message=failed.message,
                recoverable=failed.recoverable,
                last_successful_inference_time=failed.last_successful_inference_time,
                failure_count=sum(health.failure_count for health in healths),
            )
        latest = max(
            (health.last_successful_inference_time for health in healths if health.last_successful_inference_time),
            default=None,
        )
        return BackendHealth(
            state=BackendState.READY,
            ready=True,
            last_successful_inference_time=latest,
            failure_count=sum(health.failure_count for health in healths),
        )

    def reset(self, deadline: datetime | None = None) -> None:
        del deadline

    def close(self) -> None:
        return None

    @staticmethod
    def _unique(components: tuple[object, ...]) -> tuple[object, ...]:
        unique: list[object] = []
        seen: set[int] = set()
        for component in components:
            if id(component) not in seen:
                seen.add(id(component))
                unique.append(component)
        return tuple(unique)


class SequentialModelExecutor(ComponentModelExecutor):
    """Run a fixed stage sequence while delegating resources to owned components."""

    supports_priority_zero_deadline_admission = True

    def execute(self, request: ModelRequest, context: ExecutionContext) -> object:
        return self._execute_stages(request, context)

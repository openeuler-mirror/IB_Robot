"""Production functional-stage executor for independently triggered Ascend pipelines."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from copy import copy
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType

import numpy as np

from ibrobot_tracing import get_trace_emitter
from inference_service.backends.types import BackendHealth, RuntimeContext
from inference_service.pipeline.executor import ComponentModelExecutor
from inference_service.pipeline.runtime_core import ExecutionControl, StageFrame
from inference_service.pipeline.stages import InferenceStage, IterativeStage, ModelStage, ResultAdapter
from inference_service.scheduler.operations import OperationIdentity, OperationKind, OperationRegistry
from inference_service.unified_runtime import ExecutionContext, ModelRequest

trace = get_trace_emitter("ib_trace.policy.staged", component_id="policy.staged")


@dataclass(frozen=True)
class StagedScheduling:
    priorities: Mapping[str, int]
    triggers: Mapping[str, str]
    frame_base_priority: int = 0
    max_snapshot_age_ms: int = 5000


@dataclass(frozen=True)
class _VisualSnapshot:
    generation: int
    version: int
    values: Mapping[str, object]
    execution: object
    observation_mono_ns: int


class VisualFrameSupersededError(RuntimeError):
    """A pending frame was replaced before it started running."""

    operation_started = False
    outcome_known = True
    # Duck-typed marker: unified_runtime cannot import pipeline (circular).
    superseded_frame = True


class StagedModelExecutor(ComponentModelExecutor):
    """Run a frame producer and Dispatch terminal with immutable handoff.

    One worker corresponds to one persistent split-stage model instance. The
    Visual is a coalesced latest-frame producer: while one VLM run is active,
    incoming observations only replace the pending frame. When that run
    completes, the producer pulls the newest buffered frame. The terminal
    worker is called by a scheduled Dispatch. The two
    workers may overlap, while jobs targeting the same worker never overlap.
    """

    def __init__(
        self,
        stages: Iterable[InferenceStage],
        result_adapter: ResultAdapter,
        *,
        components: Iterable[object],
        execution_plan: object,
        scheduling: StagedScheduling,
        component_contexts: Mapping[int, RuntimeContext] | None = None,
        error_handler: Callable[[Exception, bool], None] | None = None,
        health_override: Callable[[], BackendHealth | None] | None = None,
        max_operation_records: int = 256,
    ) -> None:
        super().__init__(
            stages,
            result_adapter,
            components=components,
            execution_plan=execution_plan,
            component_contexts=component_contexts,
            error_handler=error_handler,
            health_override=health_override,
        )
        self._scheduling = scheduling
        if type(scheduling.max_snapshot_age_ms) is not int or scheduling.max_snapshot_age_ms <= 0:
            raise ValueError("max_snapshot_age_ms must be a positive integer")
        self._max_snapshot_age_ns = scheduling.max_snapshot_age_ms * 1_000_000
        self.frame_submitter = None
        self._generation = 1
        self._snapshot_version = 0
        self._snapshot: _VisualSnapshot | None = None
        self._snapshot_condition = threading.Condition(threading.RLock())
        self._closed = False
        self._resetting = False
        self._visual_active = 0
        self._visual_pending: tuple[object, ExecutionContext, int, Future] | None = None
        self._action_active = 0
        self._visual_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline-visual-stage")
        self._action_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline-action-stage")
        self._operations = OperationRegistry(max_records=max_operation_records, max_waiters_per_operation=4)
        self._identity = OperationIdentity(str(uuid.uuid4()), self._generation, str(uuid.uuid4()), 1, str(uuid.uuid4()))
        self._action_start = self._partition_stages()

    def load(self, context: RuntimeContext) -> None:
        deployment = context.deployment
        if getattr(deployment, "backend", None) != "ascend":
            raise ValueError("independent staged execution requires Ascend")
        model_stages = []

        def collect(stages):
            for stage in stages:
                if isinstance(stage, ModelStage):
                    model_stages.append(stage)
                elif isinstance(stage, IterativeStage):
                    collect(stage.body)

        collect(self._stages)
        for stage in model_stages:
            capabilities = getattr(stage.session, "capabilities", None)
            if (
                not getattr(capabilities, "supports_isolated_stage_execution", False)
                or getattr(capabilities, "priority_mapping", None) is None
            ):
                raise ValueError("staged execution requires loaded isolated async roles and priority streams")
        self._context = context

    def submit_frame(
        self, request: object, *, context: ExecutionContext | None = None, deadline: datetime | None = None
    ) -> Future:
        context = context or ExecutionContext(self._request_id(request), deadline)
        context.check("frame_admission")
        # ROS timestamps are converted by the node, which owns that clock.
        # Direct callers without capture metadata age from initial admission.
        request = self._stamp_observation(request)
        caller_future: Future = Future()
        caller_future.set_running_or_notify_cancel()
        pending = None
        submit_error = None
        with self._snapshot_condition:
            if self._closed:
                raise RuntimeError("staged executor is closed")
            if self._resetting:
                raise RuntimeError("staged executor is resetting")
            generation = self._generation
            if self._visual_active:
                pending = self._visual_pending
                self._visual_pending = (request, context, generation, caller_future)
            else:
                self._visual_active = 1
                try:
                    self._visual_worker.submit(self._run_visual_queue, request, context, generation, caller_future)
                except Exception as exc:
                    self._visual_active = 0
                    submit_error = exc
                    self._snapshot_condition.notify_all()
        if pending is not None:
            self._supersede_visual_future(pending[3])
        if submit_error is not None:
            caller_future.set_exception(submit_error)
        return caller_future

    def _run_visual_queue(
        self,
        request: object,
        context: ExecutionContext,
        generation: int,
        caller_future: Future,
    ) -> None:
        """Reserve the next run before callbacks can reenter producer admission."""
        while True:
            value = error = None
            try:
                value = self._run_visual(request, context, generation)
            except Exception as exc:
                error = exc
            with self._snapshot_condition:
                pending = self._visual_pending
                self._visual_pending = None
                discarded = pending if self._closed or self._resetting else None
                if discarded is not None:
                    pending = None
                self._visual_active = int(pending is not None)
                self._snapshot_condition.notify_all()
            # Future callbacks may submit another frame or take lifecycle locks.
            if discarded is not None:
                self._supersede_visual_future(discarded[3])
            if error is not None:
                caller_future.set_exception(error)
            else:
                caller_future.set_result(value)
            if pending is None:
                return
            request, context, generation, caller_future = pending

    @staticmethod
    def _supersede_visual_future(future: Future) -> None:
        if not future.done():
            future.set_exception(VisualFrameSupersededError("pending Visual frame was replaced by a newer frame"))

    @property
    def generation(self) -> int:
        with self._snapshot_condition:
            return self._generation

    def begin_generation(self) -> int:
        """Fence old Visual completions when a new product session opens."""

        with self._snapshot_condition:
            self._generation += 1
            self._snapshot = None
            self._identity = OperationIdentity(
                self._identity.session_id,
                self._generation,
                self._identity.binding_id,
                self._identity.binding_incarnation,
                self._identity.expected_boot_id,
            )
            self._snapshot_condition.notify_all()
            return self._generation

    def execute(self, request: ModelRequest, context: ExecutionContext) -> object:
        if not isinstance(request, ModelRequest):
            raise TypeError("StagedModelExecutor requires a ModelRequest")
        if not isinstance(context, ExecutionContext):
            raise TypeError("StagedModelExecutor requires an ExecutionContext")
        context.check("stage")
        request = self._stamp_observation(request)
        with self._snapshot_condition:
            if self._closed or self._resetting:
                raise RuntimeError("staged executor is unavailable")
            self._action_active += 1
        try:
            future = self._action_worker.submit(self._run_action, request, context)
        except Exception:
            with self._snapshot_condition:
                self._action_active -= 1
                self._snapshot_condition.notify_all()
            raise
        future.add_done_callback(self._action_done)
        return future.result()

    def _action_done(self, _completed: Future) -> None:
        with self._snapshot_condition:
            self._action_active = max(0, self._action_active - 1)
            self._snapshot_condition.notify_all()

    def _run_visual(self, request: object, context: ExecutionContext, generation: int) -> int:
        if not trace.enabled:
            return self._run_visual_observed(request, context, generation)
        # ThreadPoolExecutor does not inherit the caller's ContextVars. Only
        # tracing state is scoped here; the request and snapshot stay untouched.
        with (
            trace.trace_context(context.request_id, component_id="policy.visual"),
            trace.span("visual_stage", origin="built-in", generation=generation),
        ):
            version = self._run_visual_observed(request, context, generation)
            if version:
                trace.event("visual_snapshot_published", origin="built-in", generation=generation, version=version)
            return version

    def _run_visual_observed(self, request: object, context: ExecutionContext, generation: int) -> int:
        deadline = context.deadline.expires_at
        frame = self._new_frame(request, context, stage_id="visual")
        frame.values["_stage_base_priority"] = self._scheduling.frame_base_priority
        try:
            for stage in self._stages[: self._action_start]:
                context.check("stage")
                stage.execute(frame, deadline=deadline)
            context.check("frame_publish")
            snapshot = _VisualSnapshot(
                generation,
                0,
                MappingProxyType({key: self._copy_value(value) for key, value in frame.values.items()}),
                frame.execution_frame.snapshot() if frame.execution_frame is not None else None,
                request.metadata["observation_monotonic_ns"],
            )
            with self._snapshot_condition:
                if generation != self._generation:
                    return 0
                self._snapshot_version += 1
                self._snapshot = _VisualSnapshot(
                    generation,
                    self._snapshot_version,
                    snapshot.values,
                    snapshot.execution,
                    snapshot.observation_mono_ns,
                )
                self._snapshot_condition.notify_all()
                return self._snapshot_version
        except Exception as exc:
            if self._error_handler is not None:
                self._error_handler(
                    exc,
                    bool(frame.values.get("_backend_started", False)) or bool(getattr(exc, "operation_started", False)),
                )
            raise
        finally:
            frame.close()

    def _run_action(self, request: object, context: ExecutionContext) -> object:
        if not trace.enabled:
            return self._run_action_observed(request, context)
        with (
            trace.trace_context(context.request_id, component_id="policy.action"),
            trace.span("action_stage", origin="built-in"),
        ):
            return self._run_action_observed(request, context)

    def _run_action_observed(self, request: object, context: ExecutionContext) -> object:
        deadline = context.deadline.expires_at
        frame = self._new_frame(request, context, stage_id="action")
        try:
            # Build current request values without executing the visual model.
            for stage in self._stages[: self._action_start]:
                if self._stage_trigger(stage) == "frame_arrival":
                    break
                context.check("stage")
                stage.execute(frame, deadline=deadline)
            current_prompt = frame.values.get("_selected_prompt")
            snapshot = self._wait_matching_snapshot(frame.request, current_prompt, deadline, context=context)
            trace.event(
                "visual_snapshot_selected", origin="built-in", generation=snapshot.generation, version=snapshot.version
            )
            current = dict(frame.values)
            frame.values.update(snapshot.values)
            frame.values.pop("_stage_base_priority", None)
            # Current action state/prompt wins over values captured by Visual.
            frame.values.update(current)
            if frame.execution_frame is not None:
                frame.execution_frame.restore(snapshot.execution)
            for stage in self._stages[self._action_start :]:
                context.check("stage")
                stage.execute(frame, deadline=deadline)
            return self._result_adapter.adapt(frame)
        except Exception as exc:
            if self._error_handler is not None:
                self._error_handler(
                    exc,
                    bool(frame.values.get("_backend_started", False)) or bool(getattr(exc, "operation_started", False)),
                )
            raise
        finally:
            frame.close()

    def _wait_matching_snapshot(
        self,
        request: object,
        prompt: object,
        deadline: datetime | None,
        *,
        context: ExecutionContext | None = None,
    ) -> _VisualSnapshot:
        """Refresh Visual through its serialized worker when the prompt changed."""

        refresh: Future | None = None
        while True:
            if context is not None:
                context.check("snapshot")
            with self._snapshot_condition:
                snapshot = self._snapshot
                if (
                    snapshot is not None
                    and snapshot.generation == self._generation
                    and snapshot.values.get("_selected_prompt") == prompt
                ):
                    if time.monotonic_ns() - snapshot.observation_mono_ns > self._max_snapshot_age_ns:
                        snapshot = None
                    else:
                        return snapshot
                if self._closed or self._resetting:
                    raise RuntimeError("staged executor is unavailable while refreshing Visual")

            if refresh is None:
                request_id = context.request_id if context is not None else self._request_id(request)
                refresh_context = ExecutionContext(
                    f"{request_id}:refresh:{uuid.uuid4()}",
                    deadline,
                    context.cancellation_token if context is not None else None,
                )
                submit = self.frame_submitter or self.submit_frame
                refresh = submit(request, context=refresh_context)
            timeout = None
            if deadline is not None:
                timeout = max(0.0, deadline.timestamp() - time.time())
                if timeout == 0.0:
                    raise TimeoutError("action stage deadline expired refreshing Visual for the action prompt")
            try:
                refresh.result(timeout=min(0.05, timeout) if timeout is not None else 0.05)
            except VisualFrameSupersededError:
                refresh = None
                continue
            except FutureTimeoutError:
                if refresh.done():
                    # The refresh finished between the timeout and this check;
                    # re-read its outcome instead of masking it.
                    refresh.result()
                continue
            except Exception as exc:
                if isinstance(getattr(exc, "cause", None), VisualFrameSupersededError):
                    refresh = None
                    continue
                raise
            with self._snapshot_condition:
                snapshot = self._snapshot
                if (
                    snapshot is not None
                    and time.monotonic_ns() - snapshot.observation_mono_ns > self._max_snapshot_age_ns
                ):
                    # Reprocessing the same captured inputs cannot make them
                    # fresh. Avoid an unbounded producer loop on old frames.
                    raise TimeoutError("visual observation expired during refresh")
            refresh = None

    @staticmethod
    def _stamp_observation(request: object) -> ModelRequest:
        metadata = dict(getattr(request, "metadata", {}))
        if getattr(request, "request_id", None):
            metadata.setdefault("request_id", request.request_id)
        metadata.setdefault("observation_monotonic_ns", time.monotonic_ns())
        return ModelRequest(request.inputs, metadata)

    def _new_frame(self, request: object, context: ExecutionContext, *, stage_id: str) -> StageFrame:
        inputs = getattr(request, "inputs", None)
        if not isinstance(inputs, Mapping):
            raise TypeError("StagedModelExecutor request must expose mapping inputs")
        if not isinstance(request, ModelRequest):
            request = ModelRequest(inputs, getattr(request, "metadata", {}))
        frame = StageFrame(
            request,
            execution_plan=self._execution_plan,
            values={**inputs, "_execution_context": context},
            control=ExecutionControl(context.request_id, context.cancellation_token),
        )
        frame.values["_independent_stage"] = stage_id
        frame.values["_stage_priority_offsets"] = self._scheduling.priorities
        frame.values["_operation_factory"] = lambda role: self._new_operation(context, role)
        return frame

    def _new_operation(self, context: ExecutionContext, role: str):
        request_id = context.request_id
        remaining = context.deadline.remaining_seconds()
        operation, _ = self._operations.create_or_get(
            kind=OperationKind.DISPATCH,
            idempotency_key=(self._generation, request_id, role),
            identity=self._identity,
            deadline_mono_ns=max(1, time.monotonic_ns() + int((remaining if remaining is not None else 60) * 1e9)),
            waiter_id=request_id,
        )
        return operation, self._operations

    def reset(self, deadline: datetime | ExecutionContext | None = None) -> None:
        if isinstance(deadline, ExecutionContext):
            deadline = deadline.deadline.expires_at
        with self._snapshot_condition:
            self._resetting = True
            pending = self._visual_pending
            self._visual_pending = None
        if pending is not None:
            self._supersede_visual_future(pending[3])
        try:
            with self._snapshot_condition:
                while self._visual_active or self._action_active:
                    if deadline is None:
                        self._snapshot_condition.wait()
                        continue
                    remaining = max(0.0, deadline.timestamp() - time.time())
                    if remaining == 0.0 or not self._snapshot_condition.wait(remaining):
                        raise TimeoutError("staged executor reset timed out waiting for frame-arrival stages")
            # Device execution must be fenced before UNKNOWN outcomes can be
            # discarded: drain quarantined device resources first (a failure
            # aborts the reset but leaves state intact so reset can be retried).
            for component in self._components:
                drain = getattr(component, "drain_uncertain_operations", None)
                if callable(drain):
                    drain()
            self._operations.fence_uncertain(self._identity.expected_boot_id)
            if len(self._operations):
                raise RuntimeError("staged executor reset requires all device operations to have known outcomes")
            self.begin_generation()
        finally:
            with self._snapshot_condition:
                self._resetting = False
                self._snapshot_condition.notify_all()

    def close(self) -> None:
        with self._snapshot_condition:
            if self._closed:
                return
            self._closed = True
            pending = self._visual_pending
            self._visual_pending = None
            self._snapshot_condition.notify_all()
        if pending is not None:
            self._supersede_visual_future(pending[3])
        self._visual_worker.shutdown(wait=True, cancel_futures=False)
        self._action_worker.shutdown(wait=True, cancel_futures=False)

    def _partition_stages(self) -> int:
        triggered = [
            (index, trigger)
            for index, stage in enumerate(self._stages)
            if (trigger := self._stage_trigger(stage)) is not None
        ]
        frame = [index for index, trigger in triggered if trigger == "frame_arrival"]
        action = [index for index, trigger in triggered if trigger == "dispatch"]
        if not frame or not action or max(frame) >= min(action):
            raise ValueError(
                "independent staged execution requires frame-arrival producer stages before dispatch terminal stages"
            )
        return min(action)

    def _stage_trigger(self, stage: object) -> str | None:
        roles: tuple[str, ...]
        if isinstance(stage, ModelStage):
            roles = (stage.role,)
        elif isinstance(stage, IterativeStage):
            roles = tuple(stage.loop_roles)
        else:
            return None
        triggers = {self._scheduling.triggers.get(role) for role in roles}
        if None in triggers or len(triggers) != 1:
            raise ValueError(f"model stage roles {roles!r} require one shared derived trigger")
        return next(iter(triggers))

    @staticmethod
    def _copy_value(value: object) -> object:
        if isinstance(value, np.ndarray):
            result = np.array(value, copy=True, order="C")
            result.flags.writeable = False
            return result
        if isinstance(value, Mapping):
            return MappingProxyType({key: StagedModelExecutor._copy_value(item) for key, item in value.items()})
        if isinstance(value, tuple | list):
            return tuple(StagedModelExecutor._copy_value(item) for item in value)
        if callable(getattr(value, "detach", None)) and callable(getattr(value, "clone", None)):
            return value.detach().clone()
        try:
            return copy(value)
        except Exception:  # noqa: BLE001
            return value

    @staticmethod
    def _request_id(request: object) -> str:
        request_id = (
            request.metadata.get("request_id")
            if isinstance(request, ModelRequest)
            else getattr(request, "request_id", None)
        )
        if not isinstance(request_id, str) or not request_id:
            raise TypeError("staged request must expose a non-empty request_id")
        return request_id


__all__ = ["StagedModelExecutor", "StagedScheduling"]

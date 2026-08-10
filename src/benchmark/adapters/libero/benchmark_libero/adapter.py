"""LIBERO adapter with task-scoped environment lifetime."""

from __future__ import annotations

import contextlib
import os
import threading
import time
import uuid
from typing import Any

import numpy as np

from benchmark_libero.evaluation import resolve_plan_request
from benchmark_libero.init_state_loader import InitStateLoadError, load_trusted_init_states, resolve_init_states_path
from benchmark_libero.observation_codec import convert_observation
from benchmark_libero.plan_provider import NativeLiberoPlanProvider
from benchmark_libero.plan_resolver import LiberoPlanResolver
from benchmark_libero.version_probe import probe_libero_provider
from benchmark_runtime.adapter import BenchmarkAdapter
from benchmark_runtime.io_descriptor import (
    BenchmarkIODescriptor,
    FeatureDescriptor,
    ObservationBatch,
)
from benchmark_runtime.models import (
    BenchmarkCapabilities,
    BenchmarkEnvironmentConfig,
    BenchmarkTask,
    NativeArtifact,
    ResetRequest,
    ResetResult,
    StepResult,
)
from benchmark_runtime.plan import BenchmarkPlan, BenchmarkPlanTask

_RESET_SETTLING_STEPS = 10
_SETTLING_ACTION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
_FROZEN_SUITE = "libero_10"
_FROZEN_IMAGE_SIZE = (256, 256)
_FROZEN_OBSERVATION_KEYS: tuple[str, ...] = ("agentview_image", "robot0_eye_in_hand_image")
_FROZEN_CAMERA_NAMES: tuple[str, ...] = ("agentview", "robot0_eye_in_hand")
_FROZEN_CONTROL_MODE = "relative"
_LIBERO_IO_DESCRIPTOR = BenchmarkIODescriptor(
    observations=(
        FeatureDescriptor("observation.images.image", "image", "uint8", (256, 256, 3), "HWC"),
        FeatureDescriptor("observation.images.image2", "image", "uint8", (256, 256, 3), "HWC"),
        FeatureDescriptor("observation.state", "state", "float32", (8,), "C"),
    ),
    actions=(FeatureDescriptor("action", "action", "float32", (7,), "C"),),
)


class LiberoAdapterError(RuntimeError):
    """Raised when the LIBERO adapter cannot satisfy a request."""


class LiberoAdapter(BenchmarkAdapter):
    """Concrete adapter with one native environment per active task."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._env: Any = None
        self._task_suite: Any = None
        self._task: Any = None
        self._init_states: Any = None
        self._active_task_identity: tuple[str, int] | None = None
        self._configured = False
        self._closed = False
        self._close_count = 0
        self._task_environment_generation = 0
        self._task_environment_close_count = 0
        self._plan: BenchmarkPlan | None = None
        self._suite = _FROZEN_SUITE
        self._seed = 0
        self._use_init_state_id = True
        self._init_state_id = 0
        self._task_id = 0
        self._last_raw_observation: Any = None
        self._last_raw_sim_state: Any = None
        self._last_raw_step_result: Any = None
        self._episode_transaction_id: str | None = None
        self._observation_sequence = 0
        self._capabilities = BenchmarkCapabilities(
            supports_render=False,
            supports_init_state=True,
            supports_native_artifact=True,
            supports_reward=True,
            supports_success=True,
        )

    @property
    def capabilities(self) -> BenchmarkCapabilities:
        return self._capabilities

    def get_io_descriptor(self) -> BenchmarkIODescriptor:
        """Declare the canonical LIBERO policy I/O boundary."""
        return _LIBERO_IO_DESCRIPTOR

    @property
    def active_task_identity(self) -> tuple[str, int] | None:
        with self._lock:
            return self._active_task_identity

    def configure(self, config: BenchmarkEnvironmentConfig) -> None:
        with self._lock:
            if self._configured:
                raise LiberoAdapterError("LiberoAdapter already configured")
            options = config.options
            camera_names = tuple(options.get("camera_names", _FROZEN_OBSERVATION_KEYS))
            if camera_names != _FROZEN_OBSERVATION_KEYS:
                raise LiberoAdapterError(
                    f"LIBERO evaluation requires camera_names={list(_FROZEN_OBSERVATION_KEYS)}; got {list(camera_names)}"
                )
            image_size = tuple(options.get("image_size", _FROZEN_IMAGE_SIZE))
            if image_size != _FROZEN_IMAGE_SIZE:
                raise LiberoAdapterError(
                    f"LIBERO evaluation requires image_size={list(_FROZEN_IMAGE_SIZE)}; got {list(image_size)}"
                )
            control_mode = str(options.get("control_mode", _FROZEN_CONTROL_MODE))
            if control_mode != _FROZEN_CONTROL_MODE:
                raise LiberoAdapterError(
                    f"LIBERO evaluation requires control_mode='{_FROZEN_CONTROL_MODE}'; got '{control_mode}'"
                )
            mujoco_gl = os.environ.get("MUJOCO_GL", "").lower()
            if mujoco_gl and mujoco_gl not in ("egl", "osmesa", "gbm"):
                raise LiberoAdapterError(
                    f"MUJOCO_GL='{mujoco_gl}' is not a recognized headless backend; use 'egl' or unset it"
                )

            probe_libero_provider()
            plan_request = resolve_plan_request(options)
            if plan_request is not None:
                try:
                    self._plan = LiberoPlanResolver(NativeLiberoPlanProvider()).resolve(plan_request)
                except Exception as exc:
                    raise LiberoAdapterError(f"failed to resolve LIBERO evaluation plan: {exc}") from exc
                self._validate_plan_metadata(self._plan)
                self._suite = self._plan.suite
                self._seed = self._plan.seed
                self._use_init_state_id = self._plan.init_state_policy.use_init_state_id
            else:
                # Compatibility path for configurations with evaluation disabled.
                suite = str(options.get("suite", _FROZEN_SUITE))
                if suite != _FROZEN_SUITE:
                    raise LiberoAdapterError(f"legacy disabled mode requires suite='{_FROZEN_SUITE}'; got '{suite}'")
                self._suite = suite
                self._seed = self._require_non_negative_int(options.get("seed", 0), "seed")
                self._use_init_state_id = bool(options.get("use_init_state_id", True))
                self._init_state_id = self._require_non_negative_int(options.get("init_state_id", 0), "init_state_id")
                self._task_id = self._require_non_negative_int(options.get("task_id", 0), "task_id")
            self._configured = True

    @staticmethod
    def _require_non_negative_int(value: Any, field_name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise LiberoAdapterError(f"{field_name} must be a non-negative int, got {value!r}")
        return value

    def get_plan(self) -> BenchmarkPlan:
        with self._lock:
            if self._plan is None:
                raise NotImplementedError("LIBERO evaluation is disabled; no resolved plan is available")
            return self._plan

    def list_tasks(self) -> list[BenchmarkTask]:
        with self._lock:
            if self._plan is not None:
                return [
                    BenchmarkTask(self._plan.suite, task.task_id, task.name, task.prompt) for task in self._plan.tasks
                ]
        from libero.libero import benchmark as libero_benchmark  # noqa: PLC0415

        suite = libero_benchmark.get_benchmark_dict()[self._suite]()
        return [
            BenchmarkTask(
                self._suite, task_id, str(suite.get_task(task_id).name), str(suite.get_task(task_id).language)
            )
            for task_id in range(len(suite.tasks))
        ]

    def reset(self, request: ResetRequest) -> ResetResult:
        with self._lock:
            self._require_usable()
            task_snapshot = self._validate_reset_identity(request)
            identity = (request.suite, request.task_id)
            if self._active_task_identity is not None and self._active_task_identity != identity:
                raise LiberoAdapterError(
                    f"active task {self._active_task_identity} must be finalized before switching to {identity}"
                )

            reused_environment = self._env is not None
            if self._env is None:
                self._create_task_environment(task_snapshot, request)
            elif self._active_task_identity != identity:
                raise LiberoAdapterError("active task identity is inconsistent with native environment")

            requested_id = request.init_state_id if request.use_init_state_id else 0
            init_state_count = int(len(self._init_states))
            actual_id = requested_id % init_state_count
            wrapped = requested_id != actual_id
            try:
                observations = self._reset_active_environment(request.seed, self._init_states[actual_id])
            except Exception as exc:
                # A newly-created candidate that fails initialization cannot
                # retain task identity. Existing same-task env remains owned
                # for deterministic cleanup/reset recovery.
                if self._active_task_identity == identity and getattr(self, "_candidate_initializing", False):
                    self._clear_active_environment(close=True)
                if isinstance(exc, LiberoAdapterError):
                    raise
                raise LiberoAdapterError(f"candidate environment initialization failed: {exc}") from exc
            finally:
                self._candidate_initializing = False

            self._episode_transaction_id = f"libero-{uuid.uuid4().hex}"
            self._observation_sequence = 0
            observation_batch = ObservationBatch.from_mapping(
                observations,
                episode_transaction_id=self._episode_transaction_id,
                sequence_id=0,
                capture_timestamp_ns=time.monotonic_ns(),
                clock_domain="monotonic",
            )
            return ResetResult(
                observations=observation_batch,
                task_prompt=task_snapshot.prompt,
                metadata={
                    "suite": request.suite,
                    "task_id": request.task_id,
                    "seed": request.seed,
                    "episode_index": requested_id,
                    "requested_init_state_id": requested_id,
                    "actual_init_state_id": actual_id,
                    "init_state_id": actual_id,
                    "init_state_count": init_state_count,
                    "init_state_wrapped": wrapped,
                    "task_environment_generation": self._task_environment_generation,
                    "task_environment_reused": reused_environment,
                    "task_environment_close_count": self._task_environment_close_count,
                    "settling_steps": _RESET_SETTLING_STEPS,
                    "image_size": list(_FROZEN_IMAGE_SIZE),
                    "control_mode": _FROZEN_CONTROL_MODE,
                    "camera_names": list(_FROZEN_CAMERA_NAMES),
                    "init_states_path": resolve_init_states_path(self._task, self._get_libero_path()),
                },
            )

    def _require_usable(self) -> None:
        if not self._configured:
            raise LiberoAdapterError("LiberoAdapter.reset called before configure()")
        if self._closed:
            raise LiberoAdapterError("LiberoAdapter.reset called after close()")

    def _validate_reset_identity(self, request: ResetRequest) -> BenchmarkPlanTask:
        if self._plan is None:
            if request.suite != self._suite:
                raise LiberoAdapterError(
                    f"reset request suite='{request.suite}' does not match configured suite='{self._suite}'"
                )
            if request.task_id != self._task_id:
                raise LiberoAdapterError(
                    f"reset request task_id={request.task_id} does not match configured task_id={self._task_id}"
                )
            if request.seed != self._seed:
                raise LiberoAdapterError(
                    f"reset request seed={request.seed} does not match configured seed={self._seed}"
                )
            if request.use_init_state_id != self._use_init_state_id:
                raise LiberoAdapterError(
                    f"reset request use_init_state_id={request.use_init_state_id} does not match "
                    f"configured use_init_state_id={self._use_init_state_id}"
                )
            if request.use_init_state_id and request.init_state_id != self._init_state_id:
                raise LiberoAdapterError(
                    f"reset request init_state_id={request.init_state_id} does not match "
                    f"configured init_state_id={self._init_state_id}"
                )
            native_task = self._native_task_for_legacy(request.task_id)
            return BenchmarkPlanTask(
                task_id=request.task_id,
                name=str(native_task.name),
                prompt=str(native_task.language),
                init_state_count=1,
                metadata={
                    "problem_folder": str(native_task.problem_folder),
                    "bddl_file": str(native_task.bddl_file),
                    "init_states_file": str(native_task.init_states_file),
                },
            )
        if request.suite != self._plan.suite:
            raise LiberoAdapterError(f"reset suite {request.suite!r} does not match plan suite {self._plan.suite!r}")
        if request.task_id not in self._plan.selected_task_ids:
            raise LiberoAdapterError(f"task_id {request.task_id} is not selected by the resolved plan")
        if request.seed != self._plan.seed:
            raise LiberoAdapterError(f"reset seed {request.seed} does not match configured seed {self._plan.seed}")
        if request.use_init_state_id is not True:
            raise LiberoAdapterError("LIBERO evaluation requires use_init_state_id=true")
        return self._plan.tasks[request.task_id]

    def _native_task_for_legacy(self, task_id: int) -> Any:
        from libero.libero import benchmark as libero_benchmark  # noqa: PLC0415

        suite = libero_benchmark.get_benchmark_dict()[self._suite]()
        if task_id >= len(suite.tasks):
            raise LiberoAdapterError(f"task_id {task_id} out of range")
        return suite.get_task(task_id)

    @staticmethod
    def _validate_plan_metadata(plan: BenchmarkPlan) -> None:
        """Fail before the first reset when provider resources are incomplete."""
        required = ("problem_folder", "bddl_file", "init_states_file")
        for task in plan.tasks:
            missing = [
                key
                for key in required
                if not isinstance(task.metadata.get(key), str) or not str(task.metadata.get(key)).strip()
            ]
            if missing:
                raise LiberoAdapterError(
                    f"resolved task {task.task_id} provider metadata is missing non-empty fields: {missing}"
                )

    @staticmethod
    def _get_libero_path():
        from libero.libero import get_libero_path  # noqa: PLC0415

        return get_libero_path

    def _resolve_native_task(self, snapshot: BenchmarkPlanTask) -> tuple[Any, Any]:
        from libero.libero import benchmark as libero_benchmark  # noqa: PLC0415

        problem_folder = snapshot.metadata.get("problem_folder")
        if not isinstance(problem_folder, str) or not problem_folder:
            raise LiberoAdapterError(f"resolved task {snapshot.name!r} is missing provider problem_folder metadata")
        source_suite = problem_folder if self._plan and self._plan.suite == "libero_100" else self._suite
        order = (
            self._plan.effective_task_order_index
            if source_suite in {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
            else 0
        )
        suite = libero_benchmark.get_benchmark_dict()[source_suite](task_order_index=order)
        count = suite.get_num_tasks() if hasattr(suite, "get_num_tasks") else len(suite.tasks)
        for index in range(count):
            task = suite.get_task(index)
            if str(task.name) == snapshot.name:
                return suite, task
        raise LiberoAdapterError(f"resolved task {snapshot.name!r} is missing from native suite {source_suite!r}")

    def _create_task_environment(self, snapshot: BenchmarkPlanTask, request: ResetRequest) -> None:
        from libero.libero.envs import OffScreenRenderEnv  # noqa: PLC0415

        suite, task = (
            self._resolve_native_task(snapshot)
            if self._plan is not None
            else (None, self._native_task_for_legacy(request.task_id))
        )
        init_path = resolve_init_states_path(task, self._get_libero_path())
        try:
            init_states = load_trusted_init_states(init_path)
        except InitStateLoadError as exc:
            raise LiberoAdapterError(f"failed to load trusted LIBERO init-state asset at '{init_path}': {exc}") from exc
        if init_states is None or len(init_states) == 0:
            raise LiberoAdapterError(f"suite '{request.suite}' task {request.task_id} has no init states")
        task_bddl_file = os.path.join(self._get_libero_path()("bddl_files"), task.problem_folder, task.bddl_file)
        candidate = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=_FROZEN_IMAGE_SIZE[0],
            camera_widths=_FROZEN_IMAGE_SIZE[1],
            camera_names=list(_FROZEN_CAMERA_NAMES),
        )
        self._env = candidate
        self._task_suite = suite
        self._task = task
        self._init_states = init_states
        self._active_task_identity = (request.suite, request.task_id)
        self._candidate_initializing = True
        self._task_environment_generation += 1
        print(
            "[IBROBOT_BENCHMARK][LIBERO_TASK_ENV] "
            f"event=create suite={request.suite} task={request.task_id} "
            f"generation={self._task_environment_generation} close_count={self._task_environment_close_count}",
            flush=True,
        )

    def _reset_active_environment(self, seed: int, init_state: Any) -> dict[str, np.ndarray]:
        print(
            "[IBROBOT_BENCHMARK][LIBERO_TASK_ENV] "
            f"event=reset suite={self._active_task_identity[0]} task={self._active_task_identity[1]} "
            f"generation={self._task_environment_generation} seed={seed}",
            flush=True,
        )
        self._env.seed(seed)
        raw_obs = self._env.reset()
        self._env.set_init_state(init_state)
        settling_action = np.array(_SETTLING_ACTION, dtype=np.float32)
        for _ in range(_RESET_SETTLING_STEPS):
            raw_obs, _, _, _ = self._env.step(settling_action)
        for robot in self._env.robots:
            if not bool(getattr(robot.controller, "use_delta", False)):
                robot.controller.use_delta = True
        for robot in self._env.robots:
            if not bool(getattr(robot.controller, "use_delta", False)):
                raise LiberoAdapterError("LIBERO controller.use_delta is not True after reset")
        self._last_raw_observation = raw_obs
        self._last_raw_sim_state = self._safe_sim_state()
        return convert_observation(raw_obs)

    def on_task_finalized(self, suite: str, task_id: int) -> None:
        with self._lock:
            identity = (suite, task_id)
            if self._active_task_identity is None and self._env is None:
                return
            if self._active_task_identity != identity:
                raise LiberoAdapterError(
                    f"task-finalized identity {identity} does not match active task {self._active_task_identity}"
                )
            self._clear_active_environment(close=True)

    def _clear_active_environment(self, *, close: bool) -> None:
        env = self._env
        self._env = None
        self._task_suite = None
        self._task = None
        self._init_states = None
        self._active_task_identity = None
        self._episode_transaction_id = None
        self._observation_sequence = 0
        if close and env is not None:
            with contextlib.suppress(Exception):
                env.close()
            self._task_environment_close_count += 1
            print(
                "[IBROBOT_BENCHMARK][LIBERO_TASK_ENV] "
                f"event=close generation={self._task_environment_generation} "
                f"close_count={self._task_environment_close_count}",
                flush=True,
            )

    def step(self, action: dict[str, np.ndarray]) -> StepResult:
        with self._lock:
            if not self._configured:
                raise LiberoAdapterError("LiberoAdapter.step called before configure()")
            if self._closed:
                raise LiberoAdapterError("LiberoAdapter.step called after close()")
            if self._env is None:
                raise LiberoAdapterError("LiberoAdapter.step called before reset()")
            from benchmark_libero.action_codec import validate_libero_action_dict  # noqa: PLC0415

            payload = validate_libero_action_dict(action)
            raw_obs, reward, done, info = self._env.step(payload)
            is_success = bool(self._env.check_success())
            self._last_raw_observation = raw_obs
            self._last_raw_sim_state = self._safe_sim_state()
            observations = convert_observation(raw_obs)
            if self._episode_transaction_id is None:
                raise LiberoAdapterError("LIBERO observation transaction is missing; reset required")
            self._observation_sequence += 1
            observation_batch = ObservationBatch.from_mapping(
                observations,
                episode_transaction_id=self._episode_transaction_id,
                sequence_id=self._observation_sequence,
                capture_timestamp_ns=time.monotonic_ns(),
                clock_domain="monotonic",
            )
            info_out = dict(info) if isinstance(info, dict) else {}
            info_out.update(
                task=str(self._task.name) if self._task is not None else "",
                task_id=self._active_task_identity[1] if self._active_task_identity else 0,
                done=bool(done),
                is_success=is_success,
            )
            result = StepResult(
                observations=observation_batch,
                terminated=bool(done) or is_success,
                truncated=False,
                reward=float(reward) if reward is not None else None,
                is_success=is_success,
                standard_metrics={"reward": float(reward) if reward is not None else None},
                native_metrics={"libero_done": bool(done), "libero_is_success": is_success},
                info=info_out,
            )
            self._last_raw_step_result = result
            return result

    def get_native_reset_payload(self) -> dict[str, Any] | None:
        with self._lock:
            if self._last_raw_observation is None:
                return None
            payload = {
                "observations": self._last_raw_observation,
                "native_env": self._env,
            }
            if self._last_raw_sim_state is not None:
                payload["sim_state"] = self._last_raw_sim_state
            return payload

    def get_native_step_payload(self) -> dict[str, Any] | None:
        with self._lock:
            if self._last_raw_observation is None:
                return None
            payload = {
                "observations": self._last_raw_observation,
                "native_env": self._env,
            }
            if self._last_raw_sim_state is not None:
                payload["sim_state"] = self._last_raw_sim_state
            return payload

    def _safe_sim_state(self) -> Any:
        getter = getattr(self._env, "get_sim_state", None)
        if not callable(getter):
            return None
        try:
            return getter()
        except Exception:
            return None

    def render(self) -> np.ndarray | None:
        return None

    def collect_native_artifacts(self) -> list[NativeArtifact]:
        return []

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._clear_active_environment(close=True)
            self._last_raw_observation = None
            self._last_raw_sim_state = None
            self._last_raw_step_result = None
            self._close_count += 1

    @property
    def close_count(self) -> int:
        with self._lock:
            return self._close_count

"""Launch builders for validated named inference pipelines and dispatch routing."""

from __future__ import annotations

import json

from launch_ros.actions import Node

from robot_config.benchmark_endpoints import resolve_benchmark_endpoints
from robot_config.dispatch_strategies import (
    reject_legacy_smoothing_config,
    resolve_dispatch_strategies,
    validate_executor_scheduler_pairing,
)
from robot_config.inference_config import (
    ControlModeInferenceConfig,
    InferenceConfigError,
    InferencePipelineConfig,
    parse_inference_config,
)
from robot_config.logger_utils import get_colored_logger
from robot_config.runtime_target import RuntimeTarget
from robot_config.utils import parse_bool, prepare_lerobot_env

logger = get_colored_logger("robot_config.execution")


def _resolve_use_sim_time(use_sim: object, use_sim_time: object | None = None) -> bool:
    if use_sim_time is None:
        use_sim_time = use_sim
    return parse_bool(use_sim_time, default=False)


_LEGACY_EXECUTOR_TYPE_ALIASES = {"action": "topic"}


def _resolve_executor_type(executor_type: object) -> tuple[object, bool]:
    """Resolve legacy executor spellings at the robot_config launch boundary.

    ``action`` was the historical name for the topic-publishing executor.
    Keep that compatibility in the configuration layer so the executor
    registry remains an exact canonical-type registry with no aliases or
    fallback behavior.

    Returns:
        The canonical executor type and whether a legacy spelling was used.
        The flag lets the launch builder preserve the old action parameters.
    """
    if not isinstance(executor_type, str) or not executor_type.strip():
        return executor_type, False
    canonical_type = _LEGACY_EXECUTOR_TYPE_ALIASES.get(executor_type)
    if canonical_type is None:
        return executor_type, False
    logger.warning(f"Legacy executor.type={executor_type!r} is mapped to {canonical_type!r} for the dispatcher")
    return canonical_type, True


def _benchmark_external_video_producer(
    runtime_target: RuntimeTarget,
    pipeline: InferencePipelineConfig,
) -> bool:
    # Benchmark distributed pipelines use the same loopback cloud role in both
    # DDS and RTP modes.  Keeping the inference topology stable makes the
    # user-facing observation_transport.mode switch change only observation
    # delivery instead of also adding/removing an inference process.
    if pipeline.execution_mode != "distributed":
        return False
    return runtime_target is RuntimeTarget.BENCHMARK


def _attention_viz_request(inference_config: dict) -> tuple[bool, str, dict]:
    """Compatibility helper for the inactive attention sidecar configuration."""

    config = inference_config.get("attention_viz", {}) or {}
    return parse_bool(config.get("enabled", False), default=False), str(config.get("mode") or "file"), config


def _validated_inference(robot_config: dict, control_mode: str) -> ControlModeInferenceConfig:
    return parse_inference_config(robot_config, control_mode)


def _selected_executor_pipeline(
    robot_config: dict,
    control_mode: str,
    inference: ControlModeInferenceConfig | None = None,
) -> InferencePipelineConfig:
    validated = inference or _validated_inference(robot_config, control_mode)
    if not validated.enabled or not validated.pipelines:
        raise InferenceConfigError(f"control mode {control_mode!r} has no enabled inference pipeline")

    mode_config = robot_config.get("control_modes", {}).get(control_mode, {})
    executor_config = mode_config.get("executor", {}) or {}
    selection = executor_config.get("inference_pipeline")
    if selection is None:
        if len(validated.pipelines) != 1:
            raise InferenceConfigError(
                f"control mode {control_mode!r} configures multiple inference pipelines; "
                "executor.inference_pipeline must select one explicitly"
            )
        return next(iter(validated.pipelines.values()))
    if not isinstance(selection, str) or not selection:
        raise InferenceConfigError("executor.inference_pipeline must be a non-empty pipeline ID")
    try:
        return validated.pipelines[selection]
    except KeyError as exc:
        raise InferenceConfigError(
            f"executor.inference_pipeline selects unknown pipeline {selection!r}; "
            f"available pipelines: {list(validated.pipelines)}"
        ) from exc


class ExecutorSchedulerPairingError(ValueError):
    """Raised when executor/scheduler pairing violates the executor/scheduler compatibility rules."""


def _validate_executor_scheduler_pairing(executor_type: str, scheduler_mode: str) -> None:
    """Generic executor/scheduler pairing guard (launch-builder side).

    Delegates to the canonical ``robot_config.dispatch_strategies``
    implementation so an illegal combination fails fast at launch-plan
    construction time, before the dispatcher node is even spawned, and the
    node-side defensive check (which imports the same canonical function)
    cannot silently diverge from this one. Unknown executor/scheduler
    strings are NOT aliased, case-folded or fallback-corrected here; they
    pass through so the registries can fail-fast with their own messages.

    Legal combinations:
    - ``topic`` + ``continuous``
    - ``benchmark`` + ``wait_for_feedback``
    """
    validate_executor_scheduler_pairing(executor_type, scheduler_mode)


class BenchmarkEvaluationError(ValueError):
    """Raised when benchmark.evaluation selection validation fails (production benchmark wiring/benchmark setup)."""


_FORBIDDEN_EVALUATION_ROUTING_KEYS = frozenset(("task_to_pipeline", "task_to_checkpoint"))

_LEGACY_EVALUATION_KEYS = frozenset(("parallel_envs", "task_id", "init_state_id", "use_init_state_id"))


def _is_finite_positive_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return float(value) > 0.0
    except (OverflowError, ValueError, TypeError):
        return False


def _validate_benchmark_evaluation(evaluation: object) -> None:
    """Benchmark evaluation schema validator.

    Validates the evaluation section against the supported
    contract. Does NOT create a RunPlan, expand episodes, query the
    provider, or pass any selection field into the action_dispatch Node
    parameters beyond the benchmark step service endpoint.
    """
    if not isinstance(evaluation, dict):
        raise BenchmarkEvaluationError("benchmark.evaluation must be a mapping")

    # 0. Reject legacy top-level evaluation authority keys that were
    # migrated into benchmark.evaluation by benchmark setup.
    for legacy_key in _LEGACY_EVALUATION_KEYS:
        if legacy_key in evaluation:
            raise BenchmarkEvaluationError(
                f"benchmark.evaluation.{legacy_key} is a legacy key; "
                "the evaluation authority has been migrated. Remove it and "
                "use the benchmark.evaluation schema fields instead."
            )

    # 1. suite: provider-owned exact identifier. The concrete adapter validates
    #    its supported suite catalog; this generic launch layer only validates
    #    the transport-safe scalar shape.
    suite = evaluation.get("suite")
    if type(suite) is not str or not suite or suite.strip() != suite:
        raise BenchmarkEvaluationError(
            "benchmark.evaluation.suite must be an exact non-empty string with no surrounding whitespace"
        )
    # 2. tasks: may be omitted (None) meaning all resolved tasks; if present,
    #    must be a non-empty list of non-negative unique ints in YAML order.
    tasks = evaluation.get("tasks")
    if tasks is None:
        pass  # omitted = all tasks in resolved suite order
    elif type(tasks) is not list or not tasks:
        raise BenchmarkEvaluationError("benchmark.evaluation.tasks must be omitted (all tasks) or a non-empty list")
    else:
        seen_ids: set[int] = set()
        for idx, task_id in enumerate(tasks):
            if type(task_id) is bool or type(task_id) is not int:
                raise BenchmarkEvaluationError(
                    f"benchmark.evaluation.tasks[{idx}] must be an exact int (bool rejected); got {type(task_id).__name__}"
                )
            if task_id < 0:
                raise BenchmarkEvaluationError(f"benchmark.evaluation.tasks[{idx}] must be non-negative; got {task_id}")
            if task_id in seen_ids:
                raise BenchmarkEvaluationError(f"benchmark.evaluation.tasks[{idx}] duplicates task id {task_id}")
            seen_ids.add(task_id)

    # 3. episodes_per_task: default 10; must be positive int; production (enabled=true) requires >= 2.
    episodes_per_task = evaluation.get("episodes_per_task", 10)
    if type(episodes_per_task) is bool or type(episodes_per_task) is not int or episodes_per_task <= 0:
        raise BenchmarkEvaluationError("benchmark.evaluation.episodes_per_task must be a positive int (bool rejected)")
    enabled = bool(evaluation.get("enabled", False))
    if enabled and episodes_per_task < 2:
        raise BenchmarkEvaluationError(
            f"benchmark.evaluation.episodes_per_task must be >= 2 in production (enabled=true); got {episodes_per_task}"
        )

    # 4. task_order_index: default 0; must be int >= 0.
    task_order_index = evaluation.get("task_order_index", 0)
    if type(task_order_index) is bool or type(task_order_index) is not int or task_order_index < 0:
        raise BenchmarkEvaluationError(
            "benchmark.evaluation.task_order_index must be a non-negative int (bool rejected)"
        )

    # 5. use_mp: must be false; num_procs: must be 1.
    use_mp = evaluation.get("use_mp", False)
    if use_mp is not False:
        raise BenchmarkEvaluationError(
            "benchmark.evaluation.use_mp must be false; parallel/multi-process execution is not supported"
        )
    num_procs = evaluation.get("num_procs", 1)
    if type(num_procs) is bool or type(num_procs) is not int or num_procs != 1:
        raise BenchmarkEvaluationError(
            "benchmark.evaluation.num_procs must be exactly 1 (bool rejected); parallel execution is not supported"
        )
    lane_count = evaluation.get("lane_count", 1)
    if type(lane_count) is bool or type(lane_count) is not int or lane_count < 1:
        raise BenchmarkEvaluationError("benchmark.evaluation.lane_count must be a positive int (bool rejected)")

    # 6. max_steps: default 600; must be positive int.
    max_steps = evaluation.get("max_steps", 600)
    if type(max_steps) is bool or type(max_steps) is not int or max_steps <= 0:
        raise BenchmarkEvaluationError("benchmark.evaluation.max_steps must be a positive int (bool rejected)")

    # 7. save_sim_states: default true; must be bool.
    save_sim_states = evaluation.get("save_sim_states", True)
    if not isinstance(save_sim_states, bool):
        raise BenchmarkEvaluationError("benchmark.evaluation.save_sim_states must be a boolean")

    # 8. Benchmark-wide observation transport mode. The loader materializes
    #    this selection into the effective contract and inference topology.
    observation_transport = evaluation.get("observation_transport")
    if observation_transport is not None:
        if not isinstance(observation_transport, dict):
            raise BenchmarkEvaluationError("benchmark.evaluation.observation_transport must be a mapping when present")
        mode = observation_transport.get("mode")
        if mode not in {"dds", "rtp"}:
            raise BenchmarkEvaluationError(
                "benchmark.evaluation.observation_transport.mode must be exactly 'dds' or 'rtp'"
            )
        if mode == "dds" and observation_transport.get("rtp") is not None:
            raise BenchmarkEvaluationError(
                "benchmark.evaluation.observation_transport.rtp is valid only when mode is 'rtp'"
            )

    # 9. timeouts: startup_timeout_sec (default 120.0) and max_duration_sec (default 600.0); finite positive.
    timeouts = evaluation.get("timeouts", {})
    if not isinstance(timeouts, dict):
        raise BenchmarkEvaluationError("benchmark.evaluation.timeouts must be a mapping when present")
    startup_timeout = timeouts.get("startup_timeout_sec", 120.0)
    if not _is_finite_positive_number(startup_timeout):
        raise BenchmarkEvaluationError(
            "benchmark.evaluation.timeouts.startup_timeout_sec must be a finite positive number"
        )
    max_duration = timeouts.get("max_duration_sec", 600.0)
    if not _is_finite_positive_number(max_duration):
        raise BenchmarkEvaluationError(
            "benchmark.evaluation.timeouts.max_duration_sec must be a finite positive number"
        )

    # 10. video: optional mapping; validate types when present.
    video = evaluation.get("video", {})
    if not isinstance(video, dict):
        raise BenchmarkEvaluationError("benchmark.evaluation.video must be a mapping when present")
    if "enabled" in video and not isinstance(video["enabled"], bool):
        raise BenchmarkEvaluationError("benchmark.evaluation.video.enabled must be a boolean")
    if "camera_name" in video:
        cam = video["camera_name"]
        if type(cam) is not str or not cam.strip():
            raise BenchmarkEvaluationError("benchmark.evaluation.video.camera_name must be a non-empty string")
    if "fps" in video:
        fps = video["fps"]
        if type(fps) is bool or type(fps) is not int or fps <= 0:
            raise BenchmarkEvaluationError("benchmark.evaluation.video.fps must be a positive int")
    if "single_video" in video and not isinstance(video["single_video"], bool):
        raise BenchmarkEvaluationError("benchmark.evaluation.video.single_video must be a boolean")

    # 11. output: optional mapping; validate types when present.
    output = evaluation.get("output", {})
    if not isinstance(output, dict):
        raise BenchmarkEvaluationError("benchmark.evaluation.output must be a mapping when present")
    if "root" in output:
        root = output["root"]
        if type(root) is not str or not root.strip():
            raise BenchmarkEvaluationError("benchmark.evaluation.output.root must be a non-empty string")
    if "run_name" in output:
        run_name = output["run_name"]
        if type(run_name) is not str or not run_name.strip():
            raise BenchmarkEvaluationError("benchmark.evaluation.output.run_name must be a non-empty string")
    if "write_native" in output and not isinstance(output["write_native"], bool):
        raise BenchmarkEvaluationError("benchmark.evaluation.output.write_native must be a boolean")
    if "write_canonical" in output and not isinstance(output["write_canonical"], bool):
        raise BenchmarkEvaluationError("benchmark.evaluation.output.write_canonical must be a boolean")

    # 12. failure_policy: optional mapping; validate types when present.
    failure_policy = evaluation.get("failure_policy", {})
    if not isinstance(failure_policy, dict):
        raise BenchmarkEvaluationError("benchmark.evaluation.failure_policy must be a mapping when present")
    if "continue_recoverable_episode_errors" in failure_policy and not isinstance(
        failure_policy["continue_recoverable_episode_errors"], bool
    ):
        raise BenchmarkEvaluationError(
            "benchmark.evaluation.failure_policy.continue_recoverable_episode_errors must be a boolean"
        )
    if "continue_video_errors" in failure_policy and not isinstance(failure_policy["continue_video_errors"], bool):
        raise BenchmarkEvaluationError("benchmark.evaluation.failure_policy.continue_video_errors must be a boolean")

    # 13. task_to_pipeline/task_to_checkpoint routing keys rejected.
    for forbidden_key in _FORBIDDEN_EVALUATION_ROUTING_KEYS:
        if forbidden_key in evaluation:
            raise BenchmarkEvaluationError(
                f"benchmark.evaluation.{forbidden_key} is forbidden; "
                "one run must select exactly one checkpoint/pipeline"
            )


def _validate_executor_no_routing_keys(executor_config: dict) -> None:
    """Reject task-to-pipeline/task-to-checkpoint routing in executor section."""
    for forbidden_key in _FORBIDDEN_EVALUATION_ROUTING_KEYS:
        if forbidden_key in executor_config:
            raise ExecutorSchedulerPairingError(
                f"executor.{forbidden_key} is forbidden; one run must select exactly one checkpoint/pipeline"
            )


def _resolve_dispatch_section(mode_config: dict) -> tuple[str, float, str | None, str | None]:
    """Read scheduler_mode, execution_timeout_sec and strategy selections.

    Defaults: ``continuous`` and ``30.0`` (matching the dispatcher node).
    ``chunking``/``blending`` stay ``None`` when absent so the strategy
    resolution can apply documented defaults.
    """
    dispatch = mode_config.get("dispatch", {}) or {}
    reject_legacy_smoothing_config(mode_config.get("executor", {}) or {})
    scheduler_mode = dispatch.get("scheduler", "continuous")
    execution_timeout_sec = float(dispatch.get("execution_timeout_sec", 30.0))
    chunking = dispatch.get("chunking")
    blending = dispatch.get("blending")
    return scheduler_mode, execution_timeout_sec, chunking, blending


def _resolve_dispatch_strategies(
    executor_config: dict,
    chunking_yaml: object,
    blending_yaml: object,
) -> tuple[str, str]:
    """Resolve the strategy names, rejecting the removed executor flag."""
    reject_legacy_smoothing_config(executor_config)
    selection = resolve_dispatch_strategies(
        chunking=chunking_yaml,
        blending=blending_yaml,
    )
    return selection.chunking, selection.blending


def _resolve_benchmark_step_service(robot_config: dict) -> str:
    """Resolve the benchmark step service endpoint through the benchmark endpoint resolver.

    Only called when ``executor.type == 'benchmark'``. Returns the verbatim
    resolved ``/benchmark/<instance_id>/step`` string; never reads customer
    full-path overrides.
    """
    endpoints = resolve_benchmark_endpoints(robot_config)
    return endpoints.step_service


def generate_inference_node(
    robot_config: dict,
    control_mode: str,
    use_sim: object = False,
    use_sim_time: object | None = None,
    runtime_target: RuntimeTarget = RuntimeTarget.HARDWARE,
) -> list[Node]:
    """Create one unified local or distributed-edge process per pipeline."""

    inference = _validated_inference(robot_config, control_mode)
    if not inference.enabled:
        return []
    robot_config_path = robot_config.get("_config_path", "")
    if not robot_config_path:
        raise ValueError("robot_config dict is missing '_config_path'; load it through robot_config.loader")

    environment = prepare_lerobot_env()
    use_sim_clock = _resolve_use_sim_time(use_sim, use_sim_time)
    is_sim = parse_bool(use_sim, default=False)
    nodes: list[Node] = []
    for pipeline in inference.pipelines.values():
        transport = pipeline.transport
        parameters = {
            "pipeline_id": pipeline.pipeline_id,
            "model_path": str(pipeline.model_path),
            "deployment": pipeline.deployment,
            "execution_mode": pipeline.execution_mode,
            "request_timeout": pipeline.request_timeout,
            "default_task": pipeline.default_task,
            "runtime_options_json": json.dumps(dict(pipeline.runtime_options), sort_keys=True, separators=(",", ":")),
            "robot_config_path": str(robot_config_path),
            "use_sim": is_sim,
            "use_sim_time": use_sim_clock,
            "node_name": transport.node_name,
            "action_server": transport.action_server,
            "reset_service": transport.reset_service,
            "health_topic": transport.health_topic,
            "action_topic": transport.action_topic,
            "request_topic": transport.request_topic or "",
            "result_topic": transport.result_topic or "",
            "heartbeat_topic": transport.heartbeat_topic or "",
            "video_descriptor_topic": transport.video_descriptor_topic or "",
            "video_status_topic": transport.video_status_topic or "",
        }
        if runtime_target is RuntimeTarget.BENCHMARK:
            parameters["external_video_producer"] = _benchmark_external_video_producer(runtime_target, pipeline)
        # Scheduled-path endpoints are passed to the pipeline node only
        # when the scheduler is enabled; the node registers the scheduled
        # action servers + serving status iff scheduler_enabled=true.
        scheduler = inference.scheduler if inference.scheduler is not None else None
        if scheduler is not None and scheduler.enable:
            parameters["scheduled_open_session"] = transport.open_session
            parameters["scheduled_dispatch"] = transport.dispatch
            parameters["scheduled_close_session"] = transport.close_session
            parameters["scheduled_serving_status"] = transport.serving_status
            parameters["runtime_policy_json"] = pipeline.runtime_policy_json or ""
            parameters["runtime_policy_fingerprint"] = pipeline.runtime_policy_fingerprint or ""
            parameters["hardware_resource_id"] = pipeline.hardware_resource_id or ""
            parameters["session_idle_timeout_ns"] = scheduler.session_idle_timeout_ns
            parameters["max_prompt_bytes"] = scheduler.max_prompt_bytes
            parameters["max_error_message_bytes"] = scheduler.max_error_message_bytes
            parameters["max_error_details_bytes"] = scheduler.max_error_details_bytes
            parameters["max_session_records"] = scheduler.max_session_records
            parameters["terminal_result_cache_entries"] = scheduler.terminal_result_cache_entries
            parameters["max_duplicate_waiters_per_request"] = scheduler.max_duplicate_waiters_per_request
            parameters["terminal_session_retention_ns"] = scheduler.terminal_session_retention_ns
            parameters["public_capacity_json"] = json.dumps(
                {wc.work_class: {"max_in_flight": wc.max_in_flight} for wc in pipeline.public_capacity.values()},
                sort_keys=True,
            )
        nodes.append(
            Node(
                package="inference_service",
                executable="pipeline_policy_node",
                name=transport.node_name,
                env=environment,
                parameters=[parameters],
                output="screen",
            )
        )
        if _benchmark_external_video_producer(runtime_target, pipeline):
            nodes.append(
                Node(
                    package="inference_service",
                    executable="pure_inference_node",
                    name=transport.cloud_node_name,
                    env=environment,
                    parameters=[
                        {
                            "pipeline_id": pipeline.pipeline_id,
                            "model_path": str(pipeline.model_path),
                            "deployment": pipeline.deployment,
                            "request_timeout": pipeline.request_timeout,
                            "runtime_options_json": json.dumps(
                                dict(pipeline.runtime_options), sort_keys=True, separators=(",", ":")
                            ),
                            "robot_config_path": str(robot_config_path),
                            "request_topic": transport.request_topic,
                            "result_topic": transport.result_topic,
                            "heartbeat_topic": transport.heartbeat_topic,
                            "video_descriptor_topic": transport.video_descriptor_topic,
                            "video_status_topic": transport.video_status_topic,
                        }
                    ],
                    output="screen",
                )
            )
            logger.info(f"Configured benchmark loopback cloud pipeline {pipeline.pipeline_id!r}")
        logger.info(
            f"Configured inference pipeline {pipeline.pipeline_id!r}: "
            f"{pipeline.model_path} deployment={pipeline.deployment} action={transport.action_server}"
        )
    return nodes


def generate_action_dispatcher_node(robot_config: dict, control_mode: str, use_sim: object = False) -> Node:
    robot_config_path = robot_config.get("_config_path", "")
    if not robot_config_path:
        raise ValueError("robot_config dict is missing '_config_path'; load it through robot_config.loader")

    inference = _validated_inference(robot_config, control_mode)
    pipeline = _selected_executor_pipeline(robot_config, control_mode, inference)
    mode_config = robot_config.get("control_modes", {}).get(control_mode, {})
    executor_config = mode_config.get("executor", {}) or {}
    robot_joints = robot_config.get("joints", {})

    # Resolve the historical ``action`` spelling at the launch boundary before
    # validating the executor/scheduler pairing and constructing the node.
    raw_executor_type = executor_config.get("type", "topic")
    executor_type, legacy_action = _resolve_executor_type(raw_executor_type)
    scheduler_mode, execution_timeout_sec, chunking_yaml, blending_yaml = _resolve_dispatch_section(mode_config)
    selection = resolve_dispatch_strategies(
        executor_type=executor_type,
        scheduler_mode=scheduler_mode,
        chunking=chunking_yaml,
        blending=blending_yaml,
    )
    executor_type, scheduler_mode = selection.executor_type, selection.scheduler_mode
    _validate_executor_no_routing_keys(executor_config)
    chunking, blending = selection.chunking, selection.blending

    # production benchmark wiring: only benchmark executor resolves a step service endpoint. For
    # all other executors the parameter must be empty string. The customer
    # never fills a full service path; only benchmark.type/instance_id drive
    # the resolver (benchmark endpoint resolver). When executor.type=benchmark, also validate the
    # checkpoint-centric evaluation selection (suite/tasks/episodes/single-lane).
    benchmark_step_service = ""
    if executor_type == "benchmark":
        benchmark_step_service = _resolve_benchmark_step_service(robot_config)
        benchmark_section = robot_config.get("benchmark", {}) or {}
        _validate_benchmark_evaluation(benchmark_section.get("evaluation", {}))

    # benchmark setup: resolve joint_names. When joints.all is an explicit empty list
    # (benchmark YAMLs with no joint consumer), omit the parameter entirely
    # so it does not serialize as an invalid ROS empty tuple. The
    # action_dispatcher_node does not declare joint_names; omitting it is
    # safe and the node uses its internal default. Non-benchmark YAMLs
    # either omit joints.all (gets the schema default) or provide a real
    # joint list, so their behavior is unchanged.
    joint_names = robot_joints.get("all", ["1", "2", "3", "4", "5", "6"])
    if not joint_names:
        joint_names = None

    dispatcher_parameters: dict[str, object] = {
        "robot_name": robot_config.get("name", "so101"),
        "queue_size": executor_config.get("queue_size", 100),
        "watermark_threshold": executor_config.get("watermark_threshold", 20),
        "min_queue_size": executor_config.get("min_queue_size", 10),
        "control_frequency": executor_config.get("control_frequency", 100.0),
        "temporal_ensemble_coeff": executor_config.get("temporal_ensemble_coeff", 0.01),
        "chunk_size": executor_config.get("chunk_size", 100),
        "smoothing_device": executor_config.get("smoothing_device", ""),
        "chunking_strategy": chunking,
        "blending_strategy": blending,
        "control_mode": control_mode,
        "interpolation_enabled": True,
        "interpolation_step": 0.1,
        "max_interpolation_time": 2.0,
        "on_inference_failure": "hold",
        "on_queue_exhausted": "hold",
        "max_inference_timeout": 1.0,
        "max_retry_attempts": 3,
        "retry_backoff_base": 0.5,
        "stale_obs_threshold_ms": 500,
        "exhaustion_timeout": 2.0,
        "joint_state_topic": "/joint_states",
        "robot_config_path": str(robot_config_path),
        "inference_action_server": pipeline.transport.action_server,
        "inference_reset_service": pipeline.transport.reset_service,
        "inference_timeout_sec": pipeline.request_timeout,
        "policy_reset_timeout_sec": executor_config.get("policy_reset_timeout_sec", 2.0),
        "inference_prompt": executor_config.get("inference_prompt", ""),
        "navigation_mode": executor_config.get("navigation_mode", False),
        "use_sim_time": parse_bool(use_sim, default=False),
        # Both modes use the same dispatcher node, but their last-mile
        # channels are intentionally disjoint: benchmark submits to
        # StepBenchmark, while the native path publishes action topics.
        "executor_mode": executor_config.get("mode", control_mode),
    }
    if executor_type == "benchmark":
        # Benchmark actions leave through StepBenchmark; native action-topic
        # parameters are intentionally not supplied in this mode.
        dispatcher_parameters.update(
            {
                "enable_dual_mode": False,
                "executor_type": executor_type,
                "scheduler_mode": scheduler_mode,
                "execution_timeout_sec": execution_timeout_sec,
                "benchmark_step_service": benchmark_step_service,
            }
        )
    else:
        # Native topic/action configurations retain their existing parameters
        # and action-topic output path, including legacy ``action`` configs.
        dispatcher_parameters.update(
            {
                "enable_dual_mode": executor_type == "topic" and not legacy_action,
                "dispatch_action_topic": "/action_dispatch/dispatch_action",
            }
        )
    if joint_names is not None:
        dispatcher_parameters["joint_names"] = joint_names

    return Node(
        package="action_dispatch",
        executable="action_dispatcher_node",
        name="action_dispatcher",
        parameters=[dispatcher_parameters],
        output="screen",
    )


def _executor_enabled(robot_config: dict, control_mode: str) -> bool:
    """Report whether this control mode wants an action dispatcher at all.

    Absent means enabled, so every configuration written before this switch
    existed keeps its current launch graph.
    """
    executor_config = robot_config.get("control_modes", {}).get(control_mode, {}).get("executor", {}) or {}
    return parse_bool(executor_config.get("enabled", True), default=True)


def generate_robot_evaluate_node(robot_config: dict, control_mode: str, use_sim: object = False) -> Node:
    robot_config_path = robot_config.get("_config_path", "")
    if not robot_config_path:
        raise ValueError("robot_config dict is missing '_config_path'; load it through robot_config.loader")
    pipeline = _selected_executor_pipeline(robot_config, control_mode)
    executor_config = robot_config.get("control_modes", {}).get(control_mode, {}).get("executor", {}) or {}
    return Node(
        package="robot_evaluate",
        executable="robot_evaluate_node",
        name="robot_evaluate",
        parameters=[
            {
                "robot_config_path": str(robot_config_path),
                "inference_action_server": pipeline.transport.action_server,
                "watermark_threshold": executor_config.get("watermark_threshold", 20),
                "enable_stable_mode": executor_config.get("enable_stable_mode", False),
                "use_sim_time": parse_bool(use_sim, default=False),
            }
        ],
        output="screen",
    )


def _scheduled_process_exit_shutdown_factory(node: object):
    """Build an OnProcessExit on_exit handler that tears down the whole scheduled
    topology when one required process exits."""
    from launch.events import Shutdown

    def _on_exit(_event, _context) -> object:
        logger.error(f"scheduled process {getattr(node, 'node_executable', '?')} exited; shutting down")
        return Shutdown(reason="scheduled process exited")

    return _on_exit


def _benchmark_evaluation_enabled(robot_config: dict) -> bool:
    benchmark = robot_config.get("benchmark", {})
    evaluation = benchmark.get("evaluation", {}) if isinstance(benchmark, dict) else {}
    return isinstance(evaluation, dict) and bool(evaluation.get("enabled", False))


def _benchmark_process_exit_shutdown_factory(node: object):
    """Build a Benchmark-only handler for the legacy execution topology."""
    from launch.actions import EmitEvent
    from launch.events import Shutdown

    def _on_exit(event, context) -> object:
        if context.is_shutdown:
            return None
        role = getattr(node, "node_executable", "unknown")
        returncode = event.returncode
        reason = f"benchmark required process {role!r} exited with return code {returncode}"
        logger.error(reason)
        return EmitEvent(event=Shutdown(reason=reason))

    return _on_exit


def generate_execution_nodes(
    robot_config: dict,
    control_mode: str = "model_inference",
    use_sim: object = False,
    use_sim_time: object | None = None,
    runtime_target: RuntimeTarget = RuntimeTarget.HARDWARE,
) -> list:
    """Generate the execution subgraph for one control mode.

    Returns a list of launch actions (Node or RegisterEventHandler wrappers).
    On the scheduled branch each required process (required pipelines, Scheduler,
    ScheduledActionDispatcher) is wrapped in OnProcessExit -> Shutdown so that a
    required process exit or readiness timeout tears down the whole scheduled
    topology. The false/absent branch returns bare Nodes as before.
    """
    from launch.actions import RegisterEventHandler
    from launch.event_handlers import OnProcessExit

    if not control_mode or control_mode == "default":
        control_mode = robot_config.get("default_control_mode", "model_inference")

    inference = _validated_inference(robot_config, control_mode)
    scheduler = inference.scheduler if inference.scheduler is not None else None

    # scheduler.enable=true selects the scheduled topology.
    if scheduler is not None and scheduler.enable:
        if not _executor_enabled(robot_config, control_mode):
            raise ValueError(
                f"control mode {control_mode!r} sets executor.enabled=false with inference.scheduler.enable=true; "
                "the scheduled topology exists to feed a dispatcher, so disable one or the other rather than "
                "leaving the switch to mean two different things"
            )
        inference_nodes = generate_inference_node(robot_config, control_mode, use_sim, use_sim_time, runtime_target)
        scheduler_node = generate_global_inference_scheduler_node(
            robot_config, control_mode, scheduler, _resolve_use_sim_time(use_sim, use_sim_time)
        )
        scheduled_dispatcher = generate_scheduled_action_dispatcher_node(
            robot_config, control_mode, scheduler, _resolve_use_sim_time(use_sim, use_sim_time)
        )
        benchmark_lifecycle = runtime_target is RuntimeTarget.BENCHMARK and _benchmark_evaluation_enabled(robot_config)
        if benchmark_lifecycle:
            # Benchmark distributed inference may add a separate loopback cloud
            # process for each pipeline. All of its execution nodes are required
            # for the evaluation/action closure.
            required_nodes = [*inference_nodes, scheduler_node, scheduled_dispatcher]
        else:
            required_pipeline_nodes = [
                node
                for node, pipeline in zip(inference_nodes, inference.pipelines.values(), strict=True)
                if pipeline.required
            ]
            # Optional pipelines may disappear without killing required serving.
            # Global readiness/routing already excludes their stale or missing status.
            required_nodes = [*required_pipeline_nodes, scheduler_node, scheduled_dispatcher]
        exit_handler_factory = (
            _benchmark_process_exit_shutdown_factory
            if benchmark_lifecycle
            else _scheduled_process_exit_shutdown_factory
        )
        exit_handlers = [
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=node,
                    on_exit=exit_handler_factory(node),
                )
            )
            for node in required_nodes
        ]
        return [*inference_nodes, scheduler_node, scheduled_dispatcher, *exit_handlers]

    # False or absent selects the unchanged legacy behavior.
    inference_nodes = generate_inference_node(robot_config, control_mode, use_sim, use_sim_time, runtime_target)
    if not _executor_enabled(robot_config, control_mode):
        # A recording-only deployment streams video but has no inference backend
        # to answer a request. Dispatching one anyway lets it time out, and the
        # timeout invalidates the distributed session, which stops the RTP
        # sender for good. The pipeline node stays because it owns that sender.
        return list(inference_nodes)
    dispatcher = generate_action_dispatcher_node(
        robot_config,
        control_mode,
        _resolve_use_sim_time(use_sim, use_sim_time),
    )
    nodes = [*inference_nodes, dispatcher]
    if runtime_target is RuntimeTarget.BENCHMARK and _benchmark_evaluation_enabled(robot_config):
        # The benchmark configuration uses the legacy dispatcher topology
        # (dispatch.scheduler is a benchmark contract, not inference.scheduler).
        # Add the same required-process policy without changing native launches.
        handlers = [
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=node,
                    on_exit=_benchmark_process_exit_shutdown_factory(node),
                )
            )
            for node in nodes
        ]
        return [*nodes, *handlers]
    return nodes


def generate_global_inference_scheduler_node(
    robot_config: dict,
    control_mode: str,
    scheduler: object,
    use_sim_time: bool,
) -> Node:
    """Independent GlobalInferenceScheduler process and single writable Scheduler."""
    robot_config_path = robot_config.get("_config_path", "")
    if not robot_config_path:
        raise ValueError("robot_config dict is missing '_config_path'; load it through robot_config.loader")
    inference = _validated_inference(robot_config, control_mode)
    # Read-only per-pipeline status and transport parameters keep Scheduler and
    # pipeline endpoint names aligned.
    pipeline_endpoints: list[dict] = []
    for pipeline in inference.pipelines.values():
        pipeline_endpoints.append(
            {
                "pipeline_id": pipeline.pipeline_id,
                "required": pipeline.required,
                "serving_status": pipeline.transport.serving_status,
                "open_session": pipeline.transport.open_session,
                "dispatch": pipeline.transport.dispatch,
                "close_session": pipeline.transport.close_session,
                "compatibility_group": pipeline.compatibility_group or "",
                "hardware_resource_id": pipeline.hardware_resource_id or "",
                "hardware_profile_fingerprint": pipeline.hardware_profile_fingerprint or "",
                "deployment_fingerprint": pipeline.validated_manifest.fingerprint,
                "runtime_policy_fingerprint": pipeline.runtime_policy_fingerprint or "",
                "profile_compatibility_fingerprint": pipeline.profile_compatibility_fingerprint or "",
                "profile_path": str(pipeline.profile_path) if pipeline.profile_path else "",
                "public_capacity": {
                    capacity.work_class: {
                        "max_in_flight": capacity.max_in_flight,
                    }
                    for capacity in pipeline.public_capacity.values()
                },
            }
        )
    return Node(
        package="inference_service",
        executable="global_inference_scheduler_node",
        name="global_inference_scheduler",
        parameters=[
            {
                "readiness_endpoint": scheduler.global_endpoints.readiness,
                "open_session_endpoint": scheduler.global_endpoints.open_session,
                "dispatch_endpoint": scheduler.global_endpoints.dispatch,
                "close_session_endpoint": scheduler.global_endpoints.close_session,
                "default_target_pipeline_id": inference.inference_pipeline or "",
                "pipelines_json": json.dumps(pipeline_endpoints, sort_keys=True),
                "default_open_timeout_ns": scheduler.default_open_timeout_ns,
                "default_request_timeout_ns": scheduler.default_request_timeout_ns,
                "status_stale_timeout_ns": scheduler.status_stale_timeout_ns,
                "clock_skew_tolerance_ns": scheduler.clock_skew_tolerance_ns,
                "goal_acceptance_timeout_ns": scheduler.goal_acceptance_timeout_ns,
                "goal_acceptance_safety_margin_ms": scheduler.goal_acceptance_safety_margin_ms,
                "dispatch_safety_margin_ms": scheduler.dispatch_safety_margin_ms,
                "dispatch_goal_contexts": scheduler.dispatch_goal_contexts,
                "lower_priority_dispatch_goal_contexts": scheduler.lower_priority_dispatch_goal_contexts,
                "session_idle_timeout_ns": scheduler.session_idle_timeout_ns,
                "profile_min_samples": scheduler.profile_min_samples,
                "profile_max_age_days": scheduler.profile_max_age_days,
                "max_product_requests_per_session": scheduler.max_product_requests_per_session,
                "terminal_result_cache_entries": scheduler.terminal_result_cache_entries,
                "max_duplicate_waiters_per_request": scheduler.max_duplicate_waiters_per_request,
                "max_prompt_bytes": scheduler.max_prompt_bytes,
                "max_fallback_pipelines": scheduler.max_fallback_pipelines,
                "max_error_message_bytes": scheduler.max_error_message_bytes,
                "max_error_details_bytes": scheduler.max_error_details_bytes,
                "terminal_session_retention_ns": scheduler.terminal_session_retention_ns,
                "max_session_records": scheduler.max_session_records,
                "default_priority": inference.inference_priority,
                "use_sim_time": use_sim_time,
            }
        ],
        output="screen",
    )


def generate_scheduled_action_dispatcher_node(
    robot_config: dict,
    control_mode: str,
    scheduler: object,
    use_sim_time: bool,
) -> Node:
    """ScheduledActionDispatcher executable. Shares node name
    `/action_dispatcher` with the legacy dispatcher; the two never coexist."""
    robot_config_path = robot_config.get("_config_path", "")
    if not robot_config_path:
        raise ValueError("robot_config dict is missing '_config_path'; load it through robot_config.loader")
    inference = _validated_inference(robot_config, control_mode)
    mode_config = robot_config.get("control_modes", {}).get(control_mode, {})
    executor_config = mode_config.get("executor", {}) or {}
    scheduler_mode, _execution_timeout_sec, chunking_yaml, blending_yaml = _resolve_dispatch_section(mode_config)
    executor_type, _legacy_action = _resolve_executor_type(executor_config.get("type"))
    selection = resolve_dispatch_strategies(
        executor_type=executor_type,
        scheduler_mode=scheduler_mode,
        chunking=chunking_yaml,
        blending=blending_yaml,
        entrypoint="scheduled",
    )
    chunking, blending = selection.chunking, selection.blending
    return Node(
        package="action_dispatch",
        executable="scheduled_action_dispatcher_node",
        name="action_dispatcher",
        parameters=[
            {
                "queue_size": executor_config.get("queue_size", 100),
                "watermark_threshold": executor_config.get("watermark_threshold", 20),
                "control_frequency": executor_config.get("control_frequency", 100.0),
                "temporal_ensemble_coeff": executor_config.get("temporal_ensemble_coeff", 0.01),
                "chunk_size": executor_config.get("chunk_size", 100),
                "smoothing_device": executor_config.get("smoothing_device", ""),
                "chunking_strategy": chunking,
                "blending_strategy": blending,
                "joint_state_topic": "/joint_states",
                "robot_config_path": str(robot_config_path),
                # Product callers only touch the Global Scheduler endpoints.
                "scheduler_readiness_endpoint": scheduler.global_endpoints.readiness,
                "open_session_endpoint": scheduler.global_endpoints.open_session,
                "dispatch_endpoint": scheduler.global_endpoints.dispatch,
                "close_session_endpoint": scheduler.global_endpoints.close_session,
                "startup_readiness_timeout_ns": scheduler.startup_readiness_timeout_ns,
                "default_open_timeout_ns": scheduler.default_open_timeout_ns,
                "default_request_timeout_ns": scheduler.default_request_timeout_ns,
                # executor.inference_* fields are consumed only by this dispatcher.
                "inference_pipeline": inference.inference_pipeline or "",
                "inference_fallback_chain": json.dumps(list(inference.inference_fallback_chain)),
                "inference_priority": inference.inference_priority,
                "inference_retry_json": json.dumps(dict(inference.inference_retry), sort_keys=True),
                "inference_prompt": executor_config.get("inference_prompt", ""),
                # navigation_mode=false opens on READY; true waits in STOPPED.
                "navigation_mode": executor_config.get("navigation_mode", False),
                "use_sim_time": use_sim_time,
            }
        ],
        output="screen",
    )

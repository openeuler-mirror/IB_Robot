"""Tracepoint description registry and deterministic resolution."""

from __future__ import annotations

from collections.abc import Iterable

from .model import SpanRecord, TraceEvent, TracepointDefinition

TracepointIdentity = tuple[str, str, str, str]

BUILTIN_TRACEPOINT_REGISTRY: dict[TracepointIdentity, str] = {
    (
        "event",
        "action_dispatcher.decode",
        "dispatch_result",
        "built-in",
    ): "记录动作分发器收到并解码的推理结果。",
    (
        "event",
        "action_dispatcher.execute",
        "action_execute",
        "built-in",
    ): "记录当前请求中采样保留的后续动作发布结果。",
    (
        "event",
        "action_dispatcher.execute",
        "action_topic_publish",
        "built-in",
    ): "记录动作数据向控制话题的发布结果。",
    (
        "event",
        "action_dispatcher.execute",
        "first_action_execute",
        "built-in",
    ): "记录当前请求首个非保持动作的发布结果。",
    (
        "event",
        "action_dispatcher.execute",
        "safe_stop_topic_publish",
        "built-in",
    ): "记录 scheduled safe-stop 向单个控制话题发布安全命令。",
    (
        "event",
        "action_dispatcher.queue",
        "queue_refill",
        "built-in",
    ): "记录新动作块写入待执行队列后的状态。",
    (
        "event",
        "action_dispatcher.request",
        "dispatch_request",
        "built-in",
    ): "记录动作分发器发起一次策略推理请求。",
    (
        "event",
        "policy",
        "dispatch_result",
        "built-in",
    ): "记录策略节点生成推理结果及其状态。",
    (
        "event",
        "policy.observation",
        "obs_frame",
        "built-in",
    ): "记录一次观测帧采样的完整度。",
    (
        "event",
        "policy.observation",
        "obs_receive",
        "built-in",
    ): "记录策略节点接收单项观测数据及其传输时延。",
    (
        "event",
        "policy.observation",
        "obs_sample",
        "built-in",
    ): "记录按请求时间采样单项观测数据的结果与数据年龄。",
    (
        "span",
        "action_dispatcher.decode",
        "dispatch_decode",
        "built-in",
    ): "解码推理结果中的动作块。",
    (
        "span",
        "action_dispatcher.execute",
        "action_execute",
        "built-in",
    ): "发布当前请求中采样保留的后续动作。",
    (
        "span",
        "action_dispatcher.execute",
        "first_action_execute",
        "built-in",
    ): "发布当前请求的首个非保持动作。",
    (
        "span",
        "action_dispatcher.queue",
        "queue_refill",
        "built-in",
    ): "使用新的动作块更新待执行动作队列。",
    (
        "span",
        "cloud_inference",
        "model_call",
        "built-in",
    ): "在云端推理节点执行一次模型前向计算。",
    (
        "span",
        "global_scheduler",
        "scheduler_dispatch",
        "built-in",
    ): "Global scheduler 对一次 scheduled 请求执行准入、选路和下游调用。",
    (
        "span",
        "policy",
        "cloud_roundtrip",
        "built-in",
    ): "等待云端推理请求返回的往返过程。",
    (
        "span",
        "policy",
        "policy_pipeline",
        "built-in",
    ): "策略节点处理一次推理请求的完整流水线。",
    (
        "span",
        "policy",
        "policy_total",
        "built-in",
    ): "策略预处理、模型推理与后处理的总过程。",
    (
        "span",
        "policy.inference",
        "model_call",
        "built-in",
    ): "在本地推理引擎中执行一次模型前向计算。",
    (
        "span",
        "policy.observation",
        "observation_sampling",
        "built-in",
    ): "按请求时间戳采样并组装模型观测帧。",
    (
        "span",
        "policy.postprocess",
        "action_chunk_publish",
        "built-in",
    ): "封装并发布策略生成的动作块。",
    (
        "span",
        "policy.postprocess",
        "postprocess",
        "built-in",
    ): "将模型输出转换为动作数据。",
    (
        "span",
        "policy.postprocess",
        "result_encoding",
        "built-in",
    ): "将 scheduled pipeline 的动作结果编码为 ROS 消息。",
    (
        "span",
        "policy.preprocess",
        "preprocess",
        "built-in",
    ): "将观测数据转换为模型输入张量。",
}

# Descriptive alias for callers that only need the text mapping.
BUILTIN_TRACEPOINT_DESCRIPTIONS = BUILTIN_TRACEPOINT_REGISTRY


def builtin_tracepoint_definitions(component_ids: set[str] | None = None) -> list[TracepointDefinition]:
    return [
        TracepointDefinition(*identity, description)
        for identity, description in sorted(BUILTIN_TRACEPOINT_REGISTRY.items())
        if component_ids is None or identity[1] in component_ids
    ]


def _observed_definitions(events: Iterable[TraceEvent], spans: Iterable[SpanRecord]) -> list[TracepointDefinition]:
    definitions = [TracepointDefinition("span", span.component_id, span.name, span.origin) for span in spans]
    definitions.extend(
        TracepointDefinition(
            "event",
            event.component_id,
            event.name,
            str(event.field("origin", "built-in")),
        )
        for event in events
        if event.name not in {"span_begin", "span_end", "flow_send", "flow_receive"}
    )
    return definitions


def resolve_tracepoint_definitions(
    runtime: Iterable[TracepointDefinition],
    topology: Iterable[TracepointDefinition],
    events: Iterable[TraceEvent],
    spans: Iterable[SpanRecord],
) -> tuple[list[TracepointDefinition], list[str]]:
    """Resolve descriptions by source precedence and report stable conflicts."""
    runtime_by_identity: dict[TracepointIdentity, set[str]] = {}
    topology_by_identity: dict[TracepointIdentity, set[str]] = {}
    identities: set[TracepointIdentity] = set()

    for definitions, target in ((runtime, runtime_by_identity), (topology, topology_by_identity)):
        for definition in definitions:
            identities.add(definition.identity)
            if definition.description:
                target.setdefault(definition.identity, set()).add(definition.description)
    identities.update(definition.identity for definition in _observed_definitions(events, spans))

    resolved = []
    warnings = []
    for identity in sorted(identities):
        candidates = (
            ("runtime", sorted(runtime_by_identity.get(identity, set()))),
            ("topology", sorted(topology_by_identity.get(identity, set()))),
            ("built-in", [BUILTIN_TRACEPOINT_REGISTRY[identity]] if identity in BUILTIN_TRACEPOINT_REGISTRY else []),
        )
        source = "empty"
        description = ""
        for candidate_source, descriptions in candidates:
            if descriptions:
                source = candidate_source
                description = descriptions[0]
                break
        distinct = sorted({item for _, descriptions in candidates for item in descriptions})
        if len(distinct) > 1:
            kind, component_id, name, origin = identity
            warnings.append(
                "Conflicting tracepoint descriptions for "
                f"({kind}, {component_id}, {name}, {origin}); selected {source}: {description!r}; "
                f"candidates={distinct!r}"
            )
        resolved.append(TracepointDefinition(*identity, description))
    return resolved, warnings

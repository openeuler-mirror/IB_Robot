"""Build the stable performance topology used by CLI and future UI adapters."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from .definitions import builtin_tracepoint_definitions
from .model import Component, DataFlowEdge, TracepointDefinition, TraceTopology


class TopologyMetadataError(ValueError):
    """Invalid optional topology content, distinct from source or I/O failures."""


def _component(component_id: str, name: str, kind: str, parent_id: str = "", **kwargs: Any) -> Component:
    return Component(component_id, name, kind, parent_id=parent_id, **kwargs)


def operation_component_id(component_id: str, trace_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", trace_name.lower()).strip("_") or "event"
    return f"{component_id}.operation.{slug}"


def _slug(value: str, fallback: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or fallback


def _endpoint_identity(base_id: str, endpoint: str, used_ids: set[str]) -> tuple[str, str]:
    parts = [part for part in endpoint.strip("/").split("/") if part]
    if parts and parts[-1] == "commands":
        parts.pop()
    identity = "_".join(parts)
    candidate = f"{base_id}.{_slug(identity, 'endpoint')}"
    suffix = 2
    while candidate in used_ids:
        candidate = f"{base_id}.{_slug(identity, 'endpoint')}.{suffix}"
        suffix += 1
    used_ids.add(candidate)
    name = " / ".join(part.replace("_", " ").title() for part in parts)
    return candidate, name or base_id.replace(".", " ").title()


def normalize_topology(topology: TraceTopology) -> TraceTopology:
    """Make legacy duplicate root endpoint IDs addressable without mutating the caller."""
    topology = deepcopy(topology)
    grouped: dict[str, list[int]] = {}
    for index, component in enumerate(topology.components):
        grouped.setdefault(component.component_id, []).append(index)
    used_ids = {component_id for component_id, indexes in grouped.items() if len(indexes) == 1}
    normalized_ids = []

    for component_id, component_indexes in grouped.items():
        if len(component_indexes) < 2:
            continue
        components = [topology.components[index] for index in component_indexes]
        kinds = {component.kind for component in components}
        if len(kinds) != 1 or not kinds <= {"data_source", "data_sink"}:
            continue
        kind = components[0].kind
        edge_indexes = [
            index
            for index, edge in enumerate(topology.edges)
            if (edge.target_id if kind == "data_sink" else edge.source_id) == component_id
        ]
        for occurrence, component_index in enumerate(component_indexes):
            edge_index = edge_indexes[occurrence] if occurrence < len(edge_indexes) else None
            endpoint = topology.edges[edge_index].name if edge_index is not None else f"endpoint_{occurrence + 1}"
            new_id, name = _endpoint_identity(component_id, endpoint, used_ids)
            topology.components[component_index] = replace(components[occurrence], component_id=new_id, name=name)
            if edge_index is None:
                continue
            edge = topology.edges[edge_index]
            if kind == "data_sink":
                edge_id = f"execute.to.{new_id}" if edge.edge_id == f"execute.to.{component_id}" else edge.edge_id
                topology.edges[edge_index] = replace(edge, edge_id=edge_id, target_id=new_id)
            else:
                edge_id = f"{new_id}.to_policy" if edge.edge_id == f"{component_id}.to_policy" else edge.edge_id
                topology.edges[edge_index] = replace(edge, edge_id=edge_id, source_id=new_id)
        normalized_ids.append(component_id)

    used_edge_ids: set[str] = set()
    for index, edge in enumerate(topology.edges):
        edge_id = edge.edge_id
        if edge_id in used_edge_ids:
            base = f"{edge.edge_id}.{_slug(f'{edge.source_id}_{edge.target_id}_{edge.name}', 'edge')}"
            edge_id = base
            suffix = 2
            while edge_id in used_edge_ids:
                edge_id = f"{base}.{suffix}"
                suffix += 1
            topology.edges[index] = replace(edge, edge_id=edge_id)
        used_edge_ids.add(edge_id)
    if normalized_ids:
        topology.metadata["normalized_duplicate_component_ids"] = sorted(normalized_ids)
    return topology


def _operation(component_id: str, trace_name: str, *, provenance: str = "declared") -> Component:
    return _component(
        operation_component_id(component_id, trace_name),
        trace_name.replace("_", " ").title(),
        "operation",
        component_id,
        provenance=provenance,
        trace_name=trace_name,
    )


def bind_robot_topology(robot_config: Mapping[str, Any], *, control_mode: str | None = None) -> TraceTopology:
    mode = control_mode or str(robot_config.get("default_control_mode", "model_inference"))
    mode_config = dict(robot_config.get("control_modes", {}).get(mode, {}))
    inference = dict(mode_config.get("inference", {}))
    scheduler = inference.get("scheduler", {})
    scheduler_enabled = isinstance(scheduler, Mapping) and scheduler.get("enable") is True
    pipelines = inference.get("pipelines", {})
    pipeline_id = "policy"
    pipeline_config: Mapping[str, Any] = inference
    pipeline_nodes: dict[str, str] = {}
    if isinstance(pipelines, Mapping) and pipelines:
        for candidate_id, candidate_config in pipelines.items():
            candidate_config = candidate_config if isinstance(candidate_config, Mapping) else {}
            candidate_transport = candidate_config.get("transport", {})
            candidate_transport = candidate_transport if isinstance(candidate_transport, Mapping) else {}
            node_name = str(candidate_transport.get("node_name", f"inference_{candidate_id}"))
            pipeline_nodes[str(candidate_id)] = f"/{node_name}"
        executor = mode_config.get("executor", {})
        selected_pipeline = executor.get("inference_pipeline") if isinstance(executor, Mapping) else None
        pipeline_id = str(selected_pipeline or next(iter(pipelines)))
        candidate = pipelines.get(pipeline_id, {})
        pipeline_config = candidate if isinstance(candidate, Mapping) else {}
    execution_mode = str(pipeline_config.get("execution_mode", "monolithic"))
    transport = pipeline_config.get("transport", {})
    transport = transport if isinstance(transport, Mapping) else {}
    policy_node_name = str(transport.get("node_name", f"inference_{pipeline_id}"))
    cloud_node_name = str(transport.get("cloud_node_name", f"inference_{pipeline_id}_cloud"))
    metadata = {"config_path": str(robot_config.get("_config_path", "")), "pipeline_id": pipeline_id}
    if scheduler_enabled:
        metadata["dispatch_path"] = "scheduled"
        metadata["pipeline_ids"] = list(pipeline_nodes)
        metadata["pipeline_nodes"] = pipeline_nodes
    topology = TraceTopology(
        robot_name=str(robot_config.get("name", "")),
        control_mode=mode,
        execution_mode=execution_mode,
        metadata=metadata,
    )
    topology.components.extend(
        [
            _component(
                "action_dispatcher",
                "Action Dispatcher",
                "ros_node",
                package="action_dispatch",
                executable="scheduled_action_dispatcher_node" if scheduler_enabled else "action_dispatcher_node",
                node="/action_dispatcher",
            ),
            _component("action_dispatcher.request", "Dispatch Request", "module", "action_dispatcher"),
            _component(
                "policy",
                "Policy Pipelines" if scheduler_enabled and len(pipeline_nodes) > 1 else "Policy Node",
                "ros_node",
                package="inference_service",
                executable="pipeline_policy_node",
                node="" if scheduler_enabled and len(pipeline_nodes) > 1 else f"/{policy_node_name}",
            ),
            _component("policy.observation", "Observation Sampling", "module", "policy"),
            _component("policy.preprocess", "Preprocessor", "module", "policy"),
            _component("policy.inference", "Inference Engine", "module", "policy"),
            _component("policy.postprocess", "Postprocessor", "module", "policy"),
            _component("action_dispatcher.decode", "Result Decode", "module", "action_dispatcher"),
            _component("action_dispatcher.queue", "Action Queue", "module", "action_dispatcher"),
            _component("action_dispatcher.execute", "Action Execute", "module", "action_dispatcher"),
        ]
    )
    topology.components.extend(
        [
            _operation("policy", "policy_pipeline"),
            _operation("policy", "policy_total"),
            _operation("policy.observation", "observation_sampling"),
            _operation("policy.preprocess", "preprocess"),
            _operation("policy.inference", "model_call"),
            _operation("policy.postprocess", "postprocess"),
            _operation("policy.postprocess", "action_chunk_publish"),
            _operation("action_dispatcher.decode", "dispatch_decode"),
            _operation("action_dispatcher.queue", "queue_refill"),
            _operation("action_dispatcher.execute", "first_action_execute"),
        ]
    )
    if execution_mode == "distributed":
        topology.components.append(
            _component(
                "cloud_inference",
                "Cloud Inference",
                "ros_node",
                package="inference_service",
                executable="pure_inference_node",
                node=f"/{cloud_node_name}",
            )
        )
    if scheduler_enabled:
        topology.components.append(
            _component(
                "global_scheduler",
                "Global Scheduler",
                "ros_node",
                package="inference_service",
                executable="global_inference_scheduler_node",
                node="/global_inference_scheduler",
            )
        )
        topology.components.extend(
            [
                _operation("global_scheduler", "scheduler_dispatch"),
                _operation("policy.postprocess", "result_encoding"),
            ]
        )
    topology.logger_to_component = {
        "ib_trace.dispatch": "action_dispatcher",
        "ib_trace.execute": "action_dispatcher.execute",
        "ib_trace.policy": "policy",
        "ib_trace.inference": "cloud_inference" if execution_mode == "distributed" else "policy.inference",
        "ib_trace.user": "user.unassigned",
    }
    if scheduler_enabled:
        topology.logger_to_component["ib_trace.scheduler"] = "global_scheduler"
    inference_target = "cloud_inference" if execution_mode == "distributed" else "policy.inference"
    pipeline_stages = [
        ("observation_to_preprocess", "policy.observation", "policy.preprocess", "internal", "flow"),
        ("preprocess_to_inference", "policy.preprocess", inference_target, "internal", "flow"),
        ("inference_to_postprocess", inference_target, "policy.postprocess", "internal", "flow"),
        ("decode_to_queue", "action_dispatcher.decode", "action_dispatcher.queue", "internal", "flow"),
        ("queue_to_execute", "action_dispatcher.queue", "action_dispatcher.execute", "queue", "flow"),
    ]
    if scheduler_enabled:
        stages = [
            (
                "scheduled_dispatch_to_scheduler",
                "action_dispatcher.request",
                "global_scheduler",
                "ScheduledDispatchInfer",
                "flow",
            ),
            (
                "scheduler_to_pipeline_dispatch",
                "global_scheduler",
                "policy.observation",
                "Pipeline ScheduledDispatchInfer",
                "flow",
            ),
            *pipeline_stages[:3],
            (
                "pipeline_result_to_scheduler",
                "policy.postprocess",
                "global_scheduler",
                "Pipeline result",
                "flow",
            ),
            (
                "scheduler_result_to_dispatcher",
                "global_scheduler",
                "action_dispatcher.decode",
                "ScheduledDispatchInfer result",
                "flow",
            ),
            *pipeline_stages[3:],
        ]
    else:
        stages = [
            ("dispatch_to_observation", "action_dispatcher.request", "policy.observation", "DispatchInfer", "flow"),
            *pipeline_stages[:3],
            ("result_to_decode", "policy.postprocess", "action_dispatcher.decode", "DispatchInfer result", "flow"),
            *pipeline_stages[3:],
        ]
    topology.edges.extend(
        DataFlowEdge(edge_id, source, target, name, kind) for edge_id, source, target, name, kind in stages
    )
    contract = dict(robot_config.get("contract", {}))
    used_component_ids = {component.component_id for component in topology.components}
    observation_specs = [spec for spec in contract.get("observations", []) if isinstance(spec, Mapping)]
    observation_key_counts: dict[str, int] = {}
    for spec in observation_specs:
        key = str(spec.get("key", spec.get("name", "observation")))
        observation_key_counts[key] = observation_key_counts.get(key, 0) + 1
    for spec in observation_specs:
        key = str(spec.get("key", spec.get("name", "observation")))
        topic = str(spec.get("topic", key))
        source_id = f"observation.{key}"
        name = key
        if observation_key_counts[key] > 1 or source_id in used_component_ids:
            source_id, name = _endpoint_identity(source_id, topic, used_component_ids)
        else:
            used_component_ids.add(source_id)
        topology.components.append(_component(source_id, name, "data_source"))
        topology.edges.append(
            DataFlowEdge(
                f"{source_id}.to_policy",
                source_id,
                "policy.observation",
                topic,
                "topic",
                str(spec.get("type", "")),
                key,
            )
        )
    action_specs = [spec for spec in contract.get("actions", []) if isinstance(spec, Mapping)]
    action_key_counts: dict[str, int] = {}
    for spec in action_specs:
        key = str(spec.get("key", spec.get("name", "action")))
        action_key_counts[key] = action_key_counts.get(key, 0) + 1
    for spec in action_specs:
        key = str(spec.get("key", spec.get("name", "action")))
        publish = spec.get("publish", {}) if isinstance(spec.get("publish"), Mapping) else {}
        topic = str(publish.get("topic", spec.get("topic", key)))
        target_id = f"action.{key}"
        name = key
        if action_key_counts[key] > 1 or target_id in used_component_ids:
            target_id, name = _endpoint_identity(target_id, topic, used_component_ids)
        else:
            used_component_ids.add(target_id)
        topology.components.append(_component(target_id, name, "data_sink"))
        topology.edges.append(
            DataFlowEdge(
                f"execute.to.{target_id}",
                "action_dispatcher.execute",
                target_id,
                topic,
                "topic",
                str(publish.get("type", spec.get("type", ""))),
                key,
            )
        )
    topology.definitions = builtin_tracepoint_definitions({component.component_id for component in topology.components})
    if not scheduler_enabled:
        scheduled_only = {"result_encoding", "safe_stop_topic_publish"}
        topology.definitions = [
            definition for definition in topology.definitions if definition.name not in scheduled_only
        ]
    return topology


def enrich_topology_with_observed_operations(
    topology: TraceTopology | None,
    spans: list,
    events: list,
) -> TraceTopology | None:
    if topology is None:
        return None
    topology = deepcopy(topology)
    known = {component.component_id for component in topology.components}
    parent_ids = set(known)
    for span in spans:
        if not span.component_id or span.component_id not in parent_ids:
            continue
        component_id = operation_component_id(span.component_id, span.name)
        if component_id not in known:
            topology.components.append(_operation(span.component_id, span.name, provenance="observed"))
            known.add(component_id)
    for event in events:
        if (
            event.schema_version <= 0
            or event.field("origin", "built-in") != "user"
            or event.name in {"span_begin", "span_end", "flow_send", "flow_receive"}
            or not event.component_id
            or event.component_id not in parent_ids
        ):
            continue
        component_id = operation_component_id(event.component_id, event.name)
        if component_id not in known:
            topology.components.append(
                _component(
                    component_id,
                    event.name.replace("_", " ").title(),
                    "instant_event",
                    event.component_id,
                    provenance="observed",
                    trace_name=event.name,
                )
            )
            known.add(component_id)
    return topology


def write_topology_manifest(path: Path, topology: TraceTopology) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = topology.to_dict()
    document["schema_version"] = 2
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def read_topology_manifest(path: Path) -> TraceTopology:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TopologyMetadataError(f"Invalid topology JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise TopologyMetadataError("Topology manifest must be an object")
    schema_version = document.get("schema_version", 1)
    if type(schema_version) not in (int, str) or schema_version not in (1, 2, "1", "2"):
        raise TopologyMetadataError("Unsupported topology manifest schema (expected 1 or 2)")
    schema_version = int(schema_version)
    for key in ("robot_name", "control_mode", "execution_mode"):
        if key in document and not isinstance(document[key], str):
            raise TopologyMetadataError(f"Topology {key} must be a string")
    for key in ("logger_to_component", "metadata"):
        if not isinstance(document.get(key, {}), dict):
            raise TopologyMetadataError(f"Topology {key} must be an object")
    if any(not isinstance(value, str) for value in document.get("logger_to_component", {}).values()):
        raise TopologyMetadataError("Topology logger_to_component values must be strings")
    records = {}
    for key, record_type in (("components", Component), ("edges", DataFlowEdge), ("tracepoints", TracepointDefinition)):
        items = document.get(key, [])
        if not isinstance(items, list):
            raise TopologyMetadataError(f"Topology {key} must be an array")
        records[key] = []
        for item in items:
            if not isinstance(item, dict) or any(not isinstance(value, str) for value in item.values()):
                raise TopologyMetadataError(f"Topology {key} entries must be objects with string fields")
            try:
                records[key].append(record_type(**item))
            except (TypeError, ValueError) as exc:
                raise TopologyMetadataError(f"Invalid topology {key} entry: {exc}") from exc
    return normalize_topology(
        TraceTopology(
            robot_name=document.get("robot_name", ""),
            control_mode=document.get("control_mode", ""),
            execution_mode=document.get("execution_mode", "monolithic"),
            components=records["components"],
            edges=records["edges"],
            logger_to_component=dict(document.get("logger_to_component", {})),
            metadata=dict(document.get("metadata", {})) | {"schema_version": schema_version},
            definitions=records["tracepoints"],
        )
    )

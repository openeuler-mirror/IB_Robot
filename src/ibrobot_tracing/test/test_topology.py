import json
from copy import deepcopy

import pytest

from ibrobot_tracing.analysis import AnalysisResult
from ibrobot_tracing.model import TraceDataset
from ibrobot_tracing.projection import GraphQuery, project_graph
from ibrobot_tracing.topology import (
    TopologyMetadataError,
    bind_robot_topology,
    read_topology_manifest,
    write_topology_manifest,
)


@pytest.mark.parametrize("execution_mode", ["monolithic", "distributed"])
def test_scheduler_absent_and_false_keep_legacy_topology(execution_mode):
    config = {
        "name": "robot",
        "default_control_mode": "model_inference",
        "control_modes": {
            "model_inference": {
                "inference": {
                    "pipelines": {"policy": {"execution_mode": execution_mode}},
                },
                "executor": {"inference_pipeline": "policy"},
            }
        },
    }
    disabled_config = deepcopy(config)
    disabled_config["control_modes"]["model_inference"]["inference"]["scheduler"] = {"enable": False}

    absent = bind_robot_topology(config)
    disabled = bind_robot_topology(disabled_config)
    components = {component.component_id: component for component in absent.components}

    assert disabled.to_dict() == absent.to_dict()
    assert absent.metadata == {"config_path": "", "pipeline_id": "policy"}
    assert components["action_dispatcher"].executable == "action_dispatcher_node"
    assert "global_scheduler" not in components
    legacy_definitions = {definition.identity for definition in absent.definitions}
    assert ("event", "action_dispatcher.execute", "safe_stop_topic_publish", "built-in") not in legacy_definitions
    assert ("span", "policy.postprocess", "result_encoding", "built-in") not in legacy_definitions
    assert {edge.edge_id for edge in absent.edges} == {
        "dispatch_to_observation",
        "observation_to_preprocess",
        "preprocess_to_inference",
        "inference_to_postprocess",
        "result_to_decode",
        "decode_to_queue",
        "queue_to_execute",
    }


def test_scheduler_enabled_adds_scheduled_components_edges_and_metadata():
    topology = bind_robot_topology(
        {
            "name": "robot",
            "default_control_mode": "model_inference",
            "control_modes": {
                "model_inference": {
                    "inference": {
                        "scheduler": {"enable": True},
                        "pipelines": {
                            "fallback": {"execution_mode": "monolithic"},
                            "selected": {"execution_mode": "monolithic"},
                        },
                    },
                    "executor": {"inference_pipeline": "selected"},
                }
            },
        }
    )

    components = {component.component_id: component for component in topology.components}
    edges = {edge.edge_id: edge for edge in topology.edges}

    assert topology.metadata == {
        "config_path": "",
        "pipeline_id": "selected",
        "dispatch_path": "scheduled",
        "pipeline_ids": ["fallback", "selected"],
        "pipeline_nodes": {
            "fallback": "/inference_fallback",
            "selected": "/inference_selected",
        },
    }
    assert components["action_dispatcher"].executable == "scheduled_action_dispatcher_node"
    assert components["global_scheduler"].kind == "ros_node"
    assert components["global_scheduler"].package == "inference_service"
    assert components["global_scheduler"].executable == "global_inference_scheduler_node"
    assert components["global_scheduler"].node == "/global_inference_scheduler"
    assert components["policy"].name == "Policy Pipelines"
    assert components["policy"].node == ""
    assert components["global_scheduler.operation.scheduler_dispatch"].parent_id == "global_scheduler"
    assert components["policy.postprocess.operation.result_encoding"].parent_id == "policy.postprocess"
    assert topology.logger_to_component["ib_trace.scheduler"] == "global_scheduler"
    assert {edge_id: (edge.source_id, edge.target_id) for edge_id, edge in edges.items()} == {
        "scheduled_dispatch_to_scheduler": ("action_dispatcher.request", "global_scheduler"),
        "scheduler_to_pipeline_dispatch": ("global_scheduler", "policy.observation"),
        "observation_to_preprocess": ("policy.observation", "policy.preprocess"),
        "preprocess_to_inference": ("policy.preprocess", "policy.inference"),
        "inference_to_postprocess": ("policy.inference", "policy.postprocess"),
        "pipeline_result_to_scheduler": ("policy.postprocess", "global_scheduler"),
        "scheduler_result_to_dispatcher": ("global_scheduler", "action_dispatcher.decode"),
        "decode_to_queue": ("action_dispatcher.decode", "action_dispatcher.queue"),
        "queue_to_execute": ("action_dispatcher.queue", "action_dispatcher.execute"),
    }
    identities = {definition.identity for definition in topology.definitions}
    assert ("event", "action_dispatcher.request", "dispatch_request", "built-in") in identities
    assert ("event", "action_dispatcher.execute", "first_action_execute", "built-in") in identities


def test_distributed_topology_routes_model_flows_through_cloud_component():
    topology = bind_robot_topology(
        {
            "name": "robot",
            "default_control_mode": "model_inference",
            "control_modes": {
                "model_inference": {
                    "inference": {
                        "pipelines": {
                            "policy": {
                                "execution_mode": "distributed",
                                "transport": {
                                    "node_name": "edge_policy",
                                    "cloud_node_name": "cloud_policy",
                                },
                            }
                        }
                    },
                    "executor": {"inference_pipeline": "policy"},
                }
            },
        }
    )

    edges = {edge.edge_id: edge for edge in topology.edges}
    components = {component.component_id: component for component in topology.components}
    assert edges["preprocess_to_inference"].target_id == "cloud_inference"
    assert edges["inference_to_postprocess"].source_id == "cloud_inference"
    assert components["policy"].executable == "pipeline_policy_node"
    assert components["policy"].node == "/edge_policy"
    assert components["cloud_inference"].node == "/cloud_policy"
    assert topology.metadata["pipeline_id"] == "policy"


def test_legacy_inference_topology_keeps_execution_mode_compatibility():
    topology = bind_robot_topology(
        {
            "name": "robot",
            "default_control_mode": "model_inference",
            "control_modes": {"model_inference": {"inference": {"execution_mode": "distributed"}}},
        }
    )

    assert topology.execution_mode == "distributed"


def test_topology_manifest_v2_and_v1_reader_compatibility(tmp_path):
    topology = bind_robot_topology({"name": "robot"})
    current = write_topology_manifest(tmp_path / "v2.json", topology)

    assert json.loads(current.read_text())["schema_version"] == 2
    assert read_topology_manifest(current).metadata["schema_version"] == 2

    legacy_document = topology.to_dict()
    legacy_document["schema_version"] = 1
    for component in legacy_document["components"]:
        component.pop("trace_name", None)
    legacy = tmp_path / "v1.json"
    legacy.write_text(json.dumps(legacy_document))

    assert read_topology_manifest(legacy).metadata["schema_version"] == 1


@pytest.mark.parametrize("schema", [1, 2, "1", "2"])
def test_reader_keeps_supported_schema_versions_and_optional_defaults(tmp_path, schema):
    manifest = tmp_path / "topology.json"
    manifest.write_text(json.dumps({"schema_version": schema}))

    topology = read_topology_manifest(manifest)

    assert topology.metadata["schema_version"] == int(schema)
    assert topology.execution_mode == "monolithic"
    assert topology.components == topology.edges == topology.definitions == []


@pytest.mark.parametrize(
    "document",
    [
        {"schema_version": True},
        {"schema_version": 1.5},
        {"schema_version": []},
        {"robot_name": []},
        {"execution_mode": None},
        {"logger_to_component": []},
        {"components": [None]},
        {"components": [{"component_id": "c", "name": "C", "kind": "module", "parent_id": []}]},
        {"components": [{"component_id": "c", "name": "C", "kind": "module", "unknown": "x"}]},
        {"tracepoints": [{"kind": "span", "component_id": "c", "name": "work", "origin": "user", "description": []}]},
    ],
)
def test_reader_classifies_bad_metadata_before_it_reaches_analysis(tmp_path, document):
    manifest = tmp_path / "topology.json"
    manifest.write_text(json.dumps(document))

    with pytest.raises(TopologyMetadataError):
        read_topology_manifest(manifest)


@pytest.mark.parametrize("error", [TypeError("bug"), ValueError("bug"), RuntimeError("bug")])
def test_reader_does_not_reclassify_unexpected_normalization_errors(tmp_path, monkeypatch, error):
    manifest = tmp_path / "topology.json"
    manifest.write_text("{}")

    def fail_normalization(topology):
        raise error

    monkeypatch.setattr("ibrobot_tracing.topology.normalize_topology", fail_normalization)
    with pytest.raises(type(error), match="bug") as raised:
        read_topology_manifest(manifest)
    assert not isinstance(raised.value, TopologyMetadataError)


def test_duplicate_contract_actions_get_semantic_collision_safe_topology_ids():
    topology = bind_robot_topology(
        {
            "name": "robot",
            "contract": {
                "actions": [
                    {
                        "key": "action",
                        "publish": {"topic": "/arm_position_controller/commands"},
                    },
                    {
                        "key": "action",
                        "publish": {"topic": "/gripper_position_controller/commands"},
                    },
                ]
            },
        }
    )

    sinks = [component for component in topology.components if component.kind == "data_sink"]
    edges = [edge for edge in topology.edges if edge.contract_key == "action"]

    assert len({sink.component_id for sink in sinks}) == len(sinks) == 2
    assert len({sink.name for sink in sinks}) == len(sinks) == 2
    assert {sink.name for sink in sinks} == {"Arm Position Controller", "Gripper Position Controller"}
    assert len({edge.edge_id for edge in edges}) == len(edges) == 2
    assert {edge.target_id for edge in edges} == {sink.component_id for sink in sinks}
    assert {edge.contract_key for edge in edges} == {"action"}


def test_duplicate_contract_observations_get_semantic_collision_safe_topology_ids():
    topology = bind_robot_topology(
        {
            "name": "robot",
            "contract": {
                "observations": [
                    {"key": "observation.state", "topic": "/odom"},
                    {"key": "observation.state", "topic": "/joint_states"},
                ]
            },
        }
    )

    sources = [component for component in topology.components if component.kind == "data_source"]
    edges = [edge for edge in topology.edges if edge.contract_key == "observation.state"]

    assert len({source.component_id for source in sources}) == len(sources) == 2
    assert {source.name for source in sources} == {"Odom", "Joint States"}
    assert len({edge.edge_id for edge in edges}) == len(edges) == 2
    assert {edge.source_id for edge in edges} == {source.component_id for source in sources}


def test_reader_normalizes_legacy_duplicate_data_sink_ids(tmp_path):
    manifest = tmp_path / "legacy-duplicate-sinks.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "components": [
                    {"component_id": "executor", "name": "Executor", "kind": "module"},
                    {"component_id": "action.action", "name": "action", "kind": "data_sink"},
                    {"component_id": "action.action", "name": "action", "kind": "data_sink"},
                ],
                "edges": [
                    {
                        "edge_id": "execute.to.action.action",
                        "source_id": "executor",
                        "target_id": "action.action",
                        "name": "/arm_position_controller/commands",
                        "kind": "topic",
                        "contract_key": "action",
                    },
                    {
                        "edge_id": "execute.to.action.action",
                        "source_id": "executor",
                        "target_id": "action.action",
                        "name": "/gripper_position_controller/commands",
                        "kind": "topic",
                        "contract_key": "action",
                    },
                ],
            }
        )
    )

    topology = read_topology_manifest(manifest)
    sinks = [component for component in topology.components if component.kind == "data_sink"]

    assert len({sink.component_id for sink in sinks}) == 2
    assert len({edge.target_id for edge in topology.edges}) == 2
    assert len({edge.edge_id for edge in topology.edges}) == 2
    assert topology.metadata["normalized_duplicate_component_ids"] == ["action.action"]

    graph = project_graph(AnalysisResult(TraceDataset(), topology=topology), GraphQuery(view="components"))
    assert len([node for node in graph.nodes if node.kind == "data_sink"]) == 2

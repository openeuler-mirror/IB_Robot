"""Real rclpy parameter export/reload, with transport endpoints isolated."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import rclpy
import yaml
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter

from action_dispatch.action_dispatcher_node import ActionDispatcherNode
from action_dispatch.active_plan import PlanSource
from action_dispatch.chunk_planning import ChunkPlan
from action_dispatch.scheduled_action_dispatcher_node import ScheduledActionDispatcherNode


@pytest.fixture(params=[ActionDispatcherNode, ScheduledActionDispatcherNode])
def node_type(request, monkeypatch):
    cls = request.param
    for method in ("create_service", "create_client", "create_publisher", "create_subscription", "create_timer"):
        monkeypatch.setattr(cls, method, Mock())
    monkeypatch.setattr("action_dispatch.action_dispatcher_node.rclpy.action.ActionClient", Mock())
    monkeypatch.setattr("action_dispatch.action_dispatcher_node.ActionServer", Mock())
    monkeypatch.setattr("action_dispatch.action_dispatcher_node.create_executor", lambda *args: Mock())
    monkeypatch.setattr("action_dispatch.scheduled_action_dispatcher_node.TopicExecutor", Mock())

    def load_contract(node):
        node._action_specs = []
        node._robot_config = SimpleNamespace(runtime={})

    monkeypatch.setattr(ScheduledActionDispatcherNode, "_load_contract_and_plan", load_contract)
    yield cls


@pytest.mark.parametrize("blending", ["none", "temporal_ensemble"])
@pytest.mark.parametrize("toggle_count", [0, 1, 2])
@pytest.mark.parametrize("atomic", [False, True])
def test_dump_restart_roundtrip(node_type, blending, toggle_count, atomic, tmp_path):
    rclpy.init(args=[])
    node = node_type(
        parameter_overrides=[
            Parameter("blending_strategy", value=blending),
            Parameter("temporal_ensemble_coeff", value=0.04),
            Parameter("chunk_size", value=23),
        ]
    )
    try:
        assert node._smoothing_enabled == (blending == "temporal_ensemble")
        for _ in range(toggle_count):
            node._toggle_smoothing_cb(None, Mock())
        # Legacy startup without a manager intentionally cannot enable fusion.
        can_toggle = blending == "temporal_ensemble" or node_type is ScheduledActionDispatcherNode
        expected = blending
        if can_toggle and toggle_count % 2:
            expected = "none" if blending == "temporal_ensemble" else "temporal_ensemble"
        assert node._strategy_selection.blending == expected
        selection = node._strategy_selection
        plan = node._active_plan.snapshot()
        for attempted in ("rtc", "none", "temporal_ensemble"):
            parameters = [Parameter("blending_strategy", value=attempted)]
            result = node.set_parameters_atomically(parameters) if atomic else node.set_parameters(parameters)[0]
            assert not result.successful
            assert "toggle_smoothing" in result.reason
            assert node._strategy_selection is selection
            assert node._active_plan.snapshot() == plan
        parameters = node.get_parameters_by_prefix("")
        assert "temporal_smoothing_enabled" not in parameters
        exported = {name: parameter.value for name, parameter in parameters.items()}
        assert exported["blending_strategy"] == expected
        assert node._smoothing_enabled == (expected == "temporal_ensemble")
        path = tmp_path / "parameters.yaml"
        path.write_text(yaml.safe_dump({node.get_fully_qualified_name(): {"ros__parameters": exported}}))
    finally:
        Node.destroy_node(node)
        rclpy.shutdown()
    rclpy.init(args=["--ros-args", "--params-file", str(path)])
    restarted = node_type()
    try:
        assert restarted.get_parameter("blending_strategy").value == expected
        assert restarted._strategy_selection.blending == expected
        assert restarted._smoothing_enabled == (expected == "temporal_ensemble")
        assert restarted.get_parameter("temporal_ensemble_coeff").value == 0.04
        assert restarted.get_parameter("chunk_size").value == 23
        assert not restarted.has_parameter("temporal_smoothing_enabled")
    finally:
        Node.destroy_node(restarted)
        rclpy.shutdown()


@pytest.mark.parametrize("source", ["cli", "yaml"])
@pytest.mark.parametrize("enabled", [False, True])
def test_removed_override_rejected_before_declaration(node_type, source, enabled, tmp_path):
    if source == "yaml":
        path = tmp_path / "obsolete.yaml"
        path.write_text(yaml.safe_dump({"/**": {"ros__parameters": {"temporal_smoothing_enabled": enabled}}}))
        args = ["--ros-args", "--params-file", str(path)]
    else:
        args = ["--ros-args", "-p", f"temporal_smoothing_enabled:={str(enabled).lower()}"]
    rclpy.init(args=args)
    try:
        with pytest.raises(ValueError, match="temporal_smoothing_enabled.*removed"):
            node_type()
    finally:
        rclpy.shutdown()


@pytest.mark.parametrize("veto_first", [False, True])
def test_rejected_parameter_update_preserves_toggle_state(node_type, veto_first):
    rclpy.init(args=[])
    node = node_type(parameter_overrides=[Parameter("blending_strategy", value="temporal_ensemble")])
    try:
        node._active_plan.accept(ChunkPlan(np.ones((3, 2)), replenishment_watermark=0), PlanSource("retained"))
        reservation, action = node._active_plan.reserve()
        selection = node._strategy_selection
        plan = node._active_plan.snapshot()
        node.add_on_set_parameters_callback(
            lambda parameters: SetParametersResult(successful=False, reason="test veto")
        )
        if not veto_first:
            node.remove_on_set_parameters_callback(node._validate_blending_update)
            node.add_on_set_parameters_callback(node._validate_blending_update)
        node._toggle_smoothing_cb(None, Mock())
        assert node._strategy_selection is selection
        assert node._smoothing_enabled
        assert node.get_parameter("blending_strategy").value == "temporal_ensemble"
        assert node._active_plan.snapshot() == plan
        assert node._active_plan.is_current(reservation)
        np.testing.assert_array_equal(node._active_plan.reserve()[1], action)
        assert node._blending_update is None
    finally:
        Node.destroy_node(node)
        rclpy.shutdown()


@pytest.mark.parametrize("atomic", [False, True])
def test_direct_write_during_toggle_is_rejected(node_type, atomic):
    rclpy.init(args=[])
    node = node_type(parameter_overrides=[Parameter("blending_strategy", value="temporal_ensemble")])
    try:
        selection = node._strategy_selection
        plan = node._active_plan.snapshot()

        def during_toggle(parameters):
            if parameters[0].value == "none":

                def direct_write():
                    attempted = [Parameter("blending_strategy", value="temporal_ensemble")]
                    return node.set_parameters_atomically(attempted) if atomic else node.set_parameters(attempted)[0]

                with ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(direct_write).result(timeout=5)
                assert not result.successful
                assert node._strategy_selection is selection
                assert node._active_plan.snapshot() == plan
                assert node.get_parameter("blending_strategy").value == "temporal_ensemble"
            return SetParametersResult(successful=True)

        node.add_on_set_parameters_callback(during_toggle)
        node._toggle_smoothing_cb(None, Mock())
        assert node.get_parameter("blending_strategy").value == "none"
        assert node._strategy_selection.blending == "none"
        assert not node._smoothing_enabled
        node.remove_on_set_parameters_callback(during_toggle)
        node._toggle_smoothing_cb(None, Mock())
        assert node.get_parameter("blending_strategy").value == "temporal_ensemble"
        assert node._smoothing_enabled
    finally:
        Node.destroy_node(node)
        rclpy.shutdown()

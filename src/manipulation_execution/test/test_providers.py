import sys
from types import ModuleType

import pytest

from manipulation_execution.providers import (
    GRASP_GEOMETRY_FUNCTIONS,
    WRIST_GUARD_FUNCTIONS,
    load_provider,
)


def _install_fake_provider(monkeypatch, name: str, functions: list[str]) -> ModuleType:
    module = ModuleType(name)
    for function_name in functions:
        setattr(module, function_name, lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_load_provider_returns_none_when_not_declared():
    assert load_provider("target_geometry.grasp_geometry_provider", None, ("fn",)) is None
    assert load_provider("target_geometry.grasp_geometry_provider", "", ("fn",)) is None
    assert load_provider("target_geometry.grasp_geometry_provider", "   ", ("fn",)) is None


def test_load_provider_returns_imported_module(monkeypatch):
    module = _install_fake_provider(monkeypatch, "fake_grasp_provider", ["gripper_mesh_min_z"])

    assert (
        load_provider("target_geometry.grasp_geometry_provider", "fake_grasp_provider", ("gripper_mesh_min_z",))
        is module
    )


def test_load_provider_rejects_module_missing_required_functions(monkeypatch):
    _install_fake_provider(monkeypatch, "incomplete_provider", ["gripper_mesh_min_z"])

    with pytest.raises(ValueError) as excinfo:
        load_provider("target_geometry.grasp_geometry_provider", "incomplete_provider", GRASP_GEOMETRY_FUNCTIONS)

    message = str(excinfo.value)
    assert "target_geometry.grasp_geometry_provider" in message
    assert "incomplete_provider" in message
    assert "tabletop_clearance" in message
    assert "gripper_geometry_metrics_batch" in message


def test_load_provider_rejects_unimportable_module():
    with pytest.raises(ValueError) as excinfo:
        load_provider(
            "target_gripper.ik_orientation_guard.wrist_guard_provider",
            "definitely_not_a_real_provider_module",
            WRIST_GUARD_FUNCTIONS,
        )

    message = str(excinfo.value)
    assert "target_gripper.ik_orientation_guard.wrist_guard_provider" in message
    assert "definitely_not_a_real_provider_module" in message


def test_load_provider_rejects_non_callable_attribute(monkeypatch):
    module = ModuleType("attribute_provider")
    module.gripper_mesh_min_z = "not callable"
    monkeypatch.setitem(sys.modules, "attribute_provider", module)

    with pytest.raises(ValueError) as excinfo:
        load_provider("target_geometry.grasp_geometry_provider", "attribute_provider", ("gripper_mesh_min_z",))

    assert "gripper_mesh_min_z" in str(excinfo.value)

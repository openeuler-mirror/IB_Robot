import pytest

from embodied_common.workflow_contracts import (
    CanonicalWorkflowStep,
    compute_workflow_digest,
    normalize_workflow_steps,
    workflow_digest_preimage,
)


def _budget():
    return {
        "schema_version": 1,
        "started_at": {"sec": 10, "nanosec": 5},
        "deadline": {"sec": 20, "nanosec": 5},
    }


def test_workflow_digest_is_ordered_and_deterministic():
    steps = [
        CanonicalWorkflowStep(1, "open_gripper_skill"),
        CanonicalWorkflowStep(1, "recover_safe_pose"),
    ]
    kwargs = {
        "root_task_id": "task-1",
        "task_budget": _budget(),
        "expected_registry_epoch": "epoch-1",
        "expected_registry_generation": 1,
        "expected_registry_digest": "digest-1",
    }
    digest = compute_workflow_digest(workflow_steps=steps, **kwargs)
    assert digest == compute_workflow_digest(workflow_steps=steps, **kwargs)
    assert digest != compute_workflow_digest(workflow_steps=list(reversed(steps)), **kwargs)


def test_workflow_preimage_matches_frozen_contract():
    preimage = workflow_digest_preimage(
        root_task_id="task-1",
        task_budget=_budget(),
        expected_registry_epoch="epoch-1",
        expected_registry_generation=1,
        expected_registry_digest="digest-1",
        workflow_steps=[CanonicalWorkflowStep(1, "open_gripper_skill")],
    )
    assert set(preimage["task_budget"]) == {"started_at", "deadline"}
    assert set(preimage["workflow_steps"][0]) == {
        "schema_version",
        "skill_name",
        "target_name",
        "container_name",
        "place_name",
        "motion_direction",
        "motion_distance",
        "timeout_sec",
    }


def test_workflow_step_normalization_is_typed():
    steps = normalize_workflow_steps(
        [
            {
                "schema_version": 1,
                "skill_name": " move_relative_ee ",
                "motion_direction": "LEFT",
                "motion_distance": 0.05,
            }
        ]
    )
    assert steps[0].to_dict()["skill_name"] == "move_relative_ee"
    assert steps[0].to_dict()["motion_direction"] == "left"
    assert steps[0] == CanonicalWorkflowStep(1, "move_relative_ee", motion_direction="left", motion_distance=0.05)


@pytest.mark.parametrize("distance", [float("nan"), float("inf"), -0.1])
def test_workflow_step_rejects_invalid_distance(distance):
    with pytest.raises(ValueError):
        CanonicalWorkflowStep(1, "move_relative_ee", motion_distance=distance)


def test_workflow_rejects_more_than_sixteen_steps():
    with pytest.raises(ValueError, match="maximum of 16"):
        normalize_workflow_steps([CanonicalWorkflowStep(1, f"skill-{index}") for index in range(17)])


def test_workflow_v2_preserves_navigation_parameters_and_explicit_zero():
    step = CanonicalWorkflowStep(
        2,
        "nav_abs_coordinate",
        direction=" LEFT ",
        distance=1.25,
        degree=90.0,
        x=0.0,
        y=-2.5,
        yaw=-180.0,
    ).to_dict()

    assert step["schema_version"] == 2
    assert step["direction"] == "left"
    assert step["distance"] == 1.25
    assert step["degree"] == 90.0
    assert (step["has_x"], step["x"]) == (True, 0.0)
    assert (step["has_y"], step["y"]) == (True, -2.5)
    assert (step["has_yaw"], step["yaw"]) == (True, -180.0)


def test_workflow_v2_distinguishes_absent_coordinate_from_explicit_zero():
    absent, explicit_zero = normalize_workflow_steps(
        [
            {"schema_version": 2, "skill_name": "nav_abs_coordinate"},
            {"schema_version": 2, "skill_name": "nav_abs_coordinate", "x": 0.0},
        ]
    )

    assert (absent.to_dict()["has_x"], absent.to_dict()["x"]) == (False, 0.0)
    assert (explicit_zero.to_dict()["has_x"], explicit_zero.to_dict()["x"]) == (True, 0.0)


def test_workflow_v1_rejects_navigation_parameters():
    with pytest.raises(ValueError, match="schema_version 2"):
        CanonicalWorkflowStep(1, "nav_straight", direction="forward", distance=1.0)


def test_workflow_step_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown fields.*unexpected"):
        normalize_workflow_steps([{"schema_version": 1, "skill_name": "open_gripper_skill", "unexpected": True}])

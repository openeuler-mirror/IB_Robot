from __future__ import annotations

import json

import pytest

from inference_service.pi05_schedule import (
    PI05DenoisingSchedule,
    load_pi05_schedule,
    parse_pi05_schedule,
    uniform_pi05_schedule,
    write_pi05_schedule,
)


def test_uniform_schedule_round_trip(tmp_path):
    schedule = uniform_pi05_schedule(4)
    path = write_pi05_schedule(schedule, tmp_path / "schedule.json")

    assert schedule == PI05DenoisingSchedule(name="uniform", timesteps=(1.0, 0.75, 0.5, 0.25, 0.0))
    assert schedule.step_count == 4
    assert load_pi05_schedule(path) == schedule
    assert json.loads(path.read_text(encoding="utf-8")) == schedule.to_dict()
    assert [(step.index, step.timestep, step.delta) for step in schedule.iteration_steps()] == [
        (0, 1.0, -0.25),
        (1, 0.75, -0.25),
        (2, 0.5, -0.25),
        (3, 0.25, -0.25),
    ]


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"format": "other"}, "format"),
        ({"name": ""}, "name"),
        ({"algorithm": "heun"}, "algorithm"),
        ({"model_output": "action"}, "model_output"),
        ({"timesteps": [1.0]}, "at least 2"),
        ({"timesteps": [0.9, 0.0]}, "start"),
        ({"timesteps": [1.0, 0.1]}, "end"),
        ({"timesteps": [1.0, 0.5, 0.5, 0.0]}, "strictly decreasing"),
        ({"timesteps": [1.0, float("inf"), 0.0]}, "finite"),
        ({"extra": True}, "unknown fields"),
    ],
)
def test_schedule_parser_rejects_invalid_values(update, match):
    value = {
        "format": "pi05-denoising-schedule-v1",
        "name": "test",
        "algorithm": "euler",
        "model_output": "velocity",
        "timesteps": [1.0, 0.0],
    }
    value.update(update)

    with pytest.raises(ValueError, match=match):
        parse_pi05_schedule(value)


def test_schedule_loader_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "schedule.json"
    path.write_text(
        '{"format":"pi05-denoising-schedule-v1","name":"a","name":"b",'
        '"algorithm":"euler","model_output":"velocity","timesteps":[1.0,0.0]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_pi05_schedule(path)

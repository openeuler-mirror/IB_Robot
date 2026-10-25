# Copyright 2026 IB_Robot Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The public robot model the LeKiwi runtime hands to its facade.

The runtime launch renders the whole robot once and builds the interface
description from it, as the SO-101 runtime does. Without that the facade has no
``model`` and consumers cannot read joint groups, limits or frames from the
runtime.
"""

from pathlib import Path

import pytest

from robot_runtime.interface_description import build_description
from robot_runtime.launch_support import render_robot_description, resolve_profile_argument
from robot_runtime.profile import load_profile


def _description(name: str, simulated: bool) -> dict:
    path = resolve_profile_argument("lekiwi_robot", name)
    profile = load_profile(path)
    robot_description = render_robot_description(profile, path, simulated)
    # build_description takes use_sim (upstream runtime-simulation-platform:
    # one selection, one name; the simulated kwarg is gone).
    return build_description(profile, use_sim=simulated, robot_description=robot_description)


@pytest.mark.parametrize("simulated", (True, False), ids=("simulated", "physical"))
def test_nav_grasp_publishes_arm_gripper_and_wheels(simulated):
    groups = _description("lekiwi_nav_grasp", simulated)["model"]["joint_groups"]
    assert groups["base"] == ["7", "8", "9"]
    assert len(groups["arm"]) == 5 and len(groups["gripper"]) == 1
    assert groups["all"] == [*groups["arm"], *groups["gripper"], *groups["base"]]


@pytest.mark.parametrize("simulated", (True, False), ids=("simulated", "physical"))
def test_base_only_profile_publishes_wheels_only(simulated):
    groups = _description("lekiwi_lidar", simulated)["model"]["joint_groups"]
    assert groups["base"] == ["7", "8", "9"]
    assert groups["arm"] == [] and groups["gripper"] == []


def test_runtime_launch_passes_the_model_to_the_facade():
    # Source-level pin: the description must reach runtime_stack_actions,
    # otherwise the facade rebuilds one without a model.
    launch = (Path(__file__).resolve().parents[1] / "launch" / "runtime.launch.py").read_text()
    assert "robot_description=robot_description" in launch
    assert "interface_description=description" in launch

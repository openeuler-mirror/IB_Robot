"""Scene composition tests (so101-mujoco-runtime-platform tasks 2.1/2.2).

Covers the robot-agnostic composition API, the shared randomisation
placements (grid bit-compatibility with the historical keyframe grid, seeded
uniform sampling, and the issue #195 requirement that the composed model
carries no injected keyframes), the domain-keyed composed-model path, the
layout task declaration, the shared scene helpers and the dependency
direction (sim_models never imports robot_runtime).
"""

from __future__ import annotations

import ast
import importlib
import math
import pkgutil
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sim_models.scene_compiler import (
    compose_mujoco_model,
    composed_model_path,
    free_joint_states_topic,
    get_scene_layout,
    get_scene_task,
    randomization_placements,
    scene_dynamic_objects,
)

mujoco = pytest.importorskip("mujoco")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PKG_ROOT = Path(__file__).resolve().parents[1]
SCENE_MESHES = PKG_ROOT / "scenes" / "pick_banana" / "meshes"

# A minimal robot MJCF: six position joints (the arm joints the layout rest
# pose addresses) and one mesh asset (the SDF probe). The scene's
# banana/plate freejoints come from the pick_banana template include.
ROBOT_MJCF_TEMPLATE = """<mujoco model="robot_probe">
  <asset>
    <mesh name="probe_mesh" file="{meshes_dir}/plate_col_0.stl" scale="1 1 1"/>
  </asset>
  <worldbody>
    <body name="base">
      <geom type="mesh" mesh="probe_mesh"/>
      {joints_and_geoms}
    </body>
  </worldbody>
  <actuator>
    {actuators}
  </actuator>
</mujoco>
"""


@pytest.fixture
def robot_mjcf(tmp_path):
    joints = []
    actuators = []
    for j in range(1, 7):
        joints.append(f'<joint name="{j}" type="hinge" axis="0 0 1" range="-2 2" damping="1.5" armature="0.005"/>')
        actuators.append(f'<position name="{j}" joint="{j}" kp="100"/>')
    path = tmp_path / "robot.xml"
    path.write_text(
        ROBOT_MJCF_TEMPLATE.format(
            meshes_dir=SCENE_MESHES,
            joints_and_geoms="\n      ".join(joints),
            actuators="\n    ".join(actuators),
        ),
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture
def clean_domain(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "123")
    yield "123"


def _model_of(path: Path):
    return mujoco.MjModel.from_binary_path(str(path))


# ---------------------------------------------------------------------------
# 2.1: randomization_placements
# ---------------------------------------------------------------------------


def _historical_grid(banana_cfg: dict) -> list[tuple[float, float, float, float]]:
    """The pre-issue-195 keyframe grid computation, inlined verbatim."""
    x_min, x_max = banana_cfg["x_range"]
    y_min, y_max = banana_cfg["y_range"]
    z_spawn = float(banana_cfg.get("z_spawn", 0.05))
    xs = np.linspace(x_min, x_max, int(banana_cfg.get("grid_cols", 5)))
    ys = np.linspace(y_min, y_max, int(banana_cfg.get("grid_rows", 7)))
    yaws = [2.0 * math.pi * i / int(banana_cfg.get("yaw_steps", 6)) for i in range(int(banana_cfg.get("yaw_steps", 6)))]
    return [(float(x), float(y), z_spawn, float(yaw)) for x in xs for y in ys for yaw in yaws]


def test_grid_placements_are_bit_identical_to_the_historical_grid():
    layout = get_scene_layout("pick_banana")
    placements = randomization_placements(layout)
    assert placements == _historical_grid(layout["randomization"]["banana"]), (
        "grid mode must reproduce the historical keyframe grid exactly (old indices keep their meaning)"
    )


def test_grid_size_and_order_are_stable():
    layout = get_scene_layout("pick_banana")
    placements = randomization_placements(layout)
    # 5 x divisions × 7 y divisions × 6 yaw steps
    assert len(placements) == 5 * 7 * 6
    rand = layout["randomization"]["banana"]
    x_min, x_max = rand["x_range"]
    y_min, y_max = rand["y_range"]
    yaw_step = 2.0 * math.pi / 6
    # index 0 → (x_min, y_min, z_spawn, 0); the innermost loop is yaw
    assert placements[0] == pytest.approx((x_min, y_min, 0.05, 0.0), abs=1e-9)
    assert placements[1] == pytest.approx((x_min, y_min, 0.05, yaw_step), abs=1e-9)
    # after yaw_steps indices the y advances
    assert placements[6] == pytest.approx((x_min, y_min + (y_max - y_min) / 6, 0.05, 0.0), abs=1e-9)
    # the last entry is (x_max, y_max, last yaw)
    assert placements[-1] == pytest.approx((x_max, y_max, 0.05, 5 * yaw_step), abs=1e-9)


def test_placements_are_empty_without_randomization():
    assert randomization_placements({}) == []
    assert randomization_placements({"randomization": {}}) == []
    assert randomization_placements({"randomization": {"banana": {}}}) == []


def test_uniform_mode_is_deterministic_and_seeded():
    base = {
        "x_range": [-0.03, 0.10],
        "y_range": [-0.02, 0.10],
        "z_spawn": 0.05,
        "mode": "uniform",
        "seed": 42,
        "count": 25,
    }
    first = randomization_placements({"randomization": {"banana": dict(base)}})
    second = randomization_placements({"randomization": {"banana": dict(base)}})
    assert len(first) == 25
    assert first == second, "same seed + count must give the same sequence"

    # an independent reference implementation with the same seed agrees
    rng = random.Random(42)
    expected = [
        (
            rng.uniform(-0.03, 0.10),
            rng.uniform(-0.02, 0.10),
            0.05,
            rng.uniform(0.0, 2.0 * math.pi),
        )
        for _ in range(25)
    ]
    assert first == expected

    other_seed = randomization_placements({"randomization": {"banana": {**base, "seed": 43}}})
    assert other_seed != first, "a different seed changes the sequence"


def test_uniform_mode_rejects_missing_seed_or_count():
    with pytest.raises(ValueError, match="seed"):
        randomization_placements({"randomization": {"banana": {"mode": "uniform", "count": 5}}})
    with pytest.raises(ValueError, match="seed"):
        randomization_placements({"randomization": {"banana": {"mode": "uniform"}}})


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        randomization_placements({"randomization": {"banana": {"mode": "spiral", "seed": 1, "count": 2}}})


# ---------------------------------------------------------------------------
# 2.1: the shared scene helpers (issue #195)
# ---------------------------------------------------------------------------


def test_scene_dynamic_objects_lists_dynamic_bodies_only():
    assert scene_dynamic_objects("pick_banana") == ["plate", "banana"]
    assert scene_dynamic_objects("empty") == []
    assert scene_dynamic_objects("does_not_exist") == []


def test_free_joint_states_topic_is_scene_namespaced():
    assert free_joint_states_topic("pick_banana") == "/sim/pick_banana/free_joint_states"
    assert free_joint_states_topic("other_scene") == "/sim/other_scene/free_joint_states"


# ---------------------------------------------------------------------------
# 2.1: compose_mujoco_model / composed_model_path
# ---------------------------------------------------------------------------


def test_composed_model_path_is_keyed_by_domain(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "31")
    a = composed_model_path("pick_banana")
    monkeypatch.setenv("ROS_DOMAIN_ID", "32")
    b = composed_model_path("pick_banana")
    assert a != b
    assert a == Path(tempfile.gettempdir()) / "ibrobot_sim_models" / "31" / "pick_banana.mjb"
    monkeypatch.delenv("ROS_DOMAIN_ID")
    assert composed_model_path("pick_banana") == Path(tempfile.gettempdir()) / "ibrobot_sim_models" / "0" / (
        "pick_banana.mjb"
    )
    # no scene composes the robot-only model under a distinct name
    assert composed_model_path(None).name == "robot.mjb"


def test_compose_with_pick_banana_compiles_without_injected_keyframes(robot_mjcf, clean_domain):
    """Issue #195: placements travel through ResetWorld state overrides, not
    compiled-in keyframes — the composed model must carry no injected keys."""
    out = compose_mujoco_model(robot_mjcf, "pick_banana", ())
    assert out == composed_model_path("pick_banana")
    assert out.exists() and out.suffix == ".mjb"
    model = _model_of(out)
    assert model.nkey == 0, "no random_KKK/default keyframes may be injected any more"
    # the placement targets themselves are still part of the composed world
    for body in ("banana", "plate"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) >= 0
    grid = randomization_placements(get_scene_layout("pick_banana"))
    assert len(grid) == 5 * 7 * 6


def test_compose_with_empty_scene_includes_the_robot(robot_mjcf, clean_domain):
    out = compose_mujoco_model(robot_mjcf, "empty", ())
    assert out.exists() and out.suffix == ".mjb"
    model = _model_of(out)
    # the robot joints are present (the empty template now includes the robot)
    for j in range(1, 7):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(j)) >= 0


def test_compose_robot_only(robot_mjcf, clean_domain):
    out = compose_mujoco_model(robot_mjcf, None, ())
    assert out == composed_model_path(None)
    model = _model_of(out)
    for j in range(1, 7):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(j)) >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "banana_free") < 0


def test_compose_applies_sdf_to_the_given_meshes_only(robot_mjcf, clean_domain):
    plain = compose_mujoco_model(robot_mjcf, "pick_banana", ())
    plain_buffer = _model_of(plain).nbuffer
    with_sdf = compose_mujoco_model(robot_mjcf, "pick_banana", ("probe_mesh",))
    # SDF octrees are serialized into the model buffer: the probe mesh grows it
    assert _model_of(with_sdf).nbuffer > plain_buffer


# ---------------------------------------------------------------------------
# 2.2: the scene task declaration
# ---------------------------------------------------------------------------


def test_get_scene_task_returns_the_declaration():
    task = get_scene_task("pick_banana")
    assert task == {"executable": "pick_banana_task_node", "platforms": ["mujoco", "gazebo"]}


def test_get_scene_task_returns_none_without_a_declaration():
    # the empty scene has no layout.yaml at all
    assert get_scene_task("empty") is None
    assert get_scene_task("does_not_exist") is None


# ---------------------------------------------------------------------------
# Dependency direction (D3): sim_models never imports robot_runtime
# ---------------------------------------------------------------------------


def test_no_sim_models_module_imports_robot_runtime():
    import sim_models

    problems = []
    for module_info in pkgutil.walk_packages(sim_models.__path__, prefix="sim_models."):
        module = importlib.import_module(module_info.name)
        path = Path(getattr(module, "__file__", "") or "")
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "robot_runtime" or name.startswith("robot_runtime.") for name in names):
                problems.append(f"{module_info.name} imports robot_runtime")
    assert not problems, problems


def test_package_xml_declares_no_robot_runtime_dependency():
    content = (PKG_ROOT / "package.xml").read_text(encoding="utf-8")
    assert "robot_runtime" not in content

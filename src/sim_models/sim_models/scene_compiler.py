"""Scene compiler for IB-Robot simulation.

Provides runtime path resolution for scene assets using a template
placeholder mechanism.  Template files (.world.template, .xml.template)
contain two placeholders that are substituted at launch time:

  {{MESHES_DIR}}      — absolute path to the scene's meshes/ directory
  {{ROBOT_XML_PATH}}  — absolute path to the robot MJCF XML (MuJoCo only)

The substituted content is written to a temporary directory so that Gazebo
and MuJoCo can reference mesh files by their installed absolute paths.

Public API
----------
  get_scene_file(scene_name, platform) -> Path
      Return the Path to the raw template file (for inspection or tests).

  get_gazebo_world_path(scene_name) -> Path
      Generate the resolved Gazebo world file.
      Used by GazeboAdapter.start_backend() when simulation.scene is set.

  randomization_placements(layout) -> list[(x, y, z, yaw)]
      The single randomisation generator shared by both physics platforms
      (so101-mujoco-runtime-platform D9 / issue #195): randomisation index
      *k* maps to the same banana placement on MuJoCo (``ResetWorld`` state
      overrides) and on Gazebo (``SetEntityPose`` with the same index).
      ``mode: grid`` (default) is bit-identical to the historical keyframe
      grid; ``mode: uniform`` samples deterministically from ``seed``.
      Every input is validated (finite ordered ranges, positive-integer
      divisions, integer seed/count with count >= 1) — ValueError names the
      offending key and fires in the consumer's constructor.

  arm_rest_pairs(layout) -> list[(joint_name, rest_qpos)]
      The validated arm rest pairs (``arm_joint_names`` /
      ``arm_rest_qpos``): both fields arrays, unique non-empty names,
      equal-length arrays, finite values. Shared by the scene task's rest
      trajectories and the MuJoCo backend's ``ResetWorld`` joint override.

  scene_dynamic_objects(scene_name) -> list[str]
      Names of the scene's dynamic objects (``type: dynamic`` in layout.yaml).

  free_joint_states_topic(scene) -> str
      The scene-namespaced free-joint observation topic the MuJoCo
      FreeJointStatePublisherPlugin publishes to (and the scene task's
      MuJoCo backend subscribes to); the Gazebo counterpart is
      ``/sim/<scene>/object_poses``.

  composed_model_path(scene) -> Path
      The single composed-model location (D3), keyed by ``ROS_DOMAIN_ID`` so
      concurrent launches on different domains never overwrite each other's
      model. Used by the provider (writer); the scene task no longer reads
      the composed model since placements moved to ``ResetWorld`` state
      overrides (issue #195).

  compose_mujoco_model(robot_mjcf_path, scene, sdf_mesh_names) -> Path
      Robot-agnostic composition (D3): substitute the robot MJCF into the
      named scene (or compile it alone when ``scene`` is None), apply SDF
      collision to the given meshes, compile and save the ``.mjb``. Falls
      back to the ``.xml`` only when the ``mujoco`` package is missing; the
      provider treats that as fatal.

  get_scene_layout(scene_name) -> dict
      The parsed layout.yaml of a scene (or {}).

  get_scene_task(scene_name) -> dict | None
      The scene's ``task:`` declaration (executable + platforms), or None.
"""

import logging
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory

_MESHES_PLACEHOLDER = "{{MESHES_DIR}}"
_ROBOT_XML_PLACEHOLDER = "{{ROBOT_XML_PATH}}"

_PLATFORM_EXT = {
    "gazebo": ".world.template",
    "mujoco": ".xml.template",
}


def _finite_number(value, label: str) -> float:
    """A finite YAML number (bools are ints in Python but not valid here)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def _positive_int(value, label: str) -> int:
    """A strictly positive integer (rejects floats, bools and zero)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{label} must be >= 1, got {value}")
    return value


def _validated_range(value, label: str) -> tuple[float, float]:
    """A finite [min, max] pair with min <= max."""
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"{label} must be a [min, max] pair, got {value!r}")
    low = _finite_number(value[0], f"{label}[0]")
    high = _finite_number(value[1], f"{label}[1]")
    if low > high:
        raise ValueError(f"{label} must satisfy min <= max, got {value!r}")
    return low, high


def randomization_placements(layout: dict) -> list[tuple[float, float, float, float]]:
    """The single randomisation generator shared by both physics platforms (D9, issue #195).

    Index *k* of the returned list maps to the same banana placement on every
    platform: the MuJoCo backend's ``ResetWorld`` state overrides and the
    Gazebo backend's ``SetEntityPose`` both consume ``placements[k]``.

    ``randomization.banana.mode`` selects the generator:

    - ``grid`` (default, also when the key is absent): the historical
      keyframe-era grid — x outer, y middle, yaw inner, values from
      ``numpy.linspace`` — reproduced bit-identically, so existing
      randomisation indices keep their meaning.
    - ``uniform``: ``count`` poses sampled deterministically from ``seed``
      across the x/y/yaw ranges (``z`` stays ``z_spawn``); the same seed and
      count give the same sequence on every platform, so index *k* is still
      platform-independent.

    Every input is validated here so a bad layout fails in the consumer's
    constructor — before any motion is commanded —
    with a message naming the offending key: ranges must be finite ordered
    pairs, ``z_spawn`` finite, grid divisions positive integers, uniform
    ``seed``/``count`` integers (bools and floats rejected) with
    ``count >= 1``.
    """
    rand = layout.get("randomization", {}) or {}
    banana_cfg = rand.get("banana", {}) or {}
    if not banana_cfg:
        return []
    x_min, x_max = _validated_range(banana_cfg.get("x_range", [-0.03, 0.10]), "randomization.banana.x_range")
    y_min, y_max = _validated_range(banana_cfg.get("y_range", [-0.02, 0.10]), "randomization.banana.y_range")
    z_spawn = _finite_number(banana_cfg.get("z_spawn", 0.05), "randomization.banana.z_spawn")
    mode = str(banana_cfg.get("mode", "grid")).lower()
    if mode == "grid":
        grid_cols = _positive_int(banana_cfg.get("grid_cols", 5), "randomization.banana.grid_cols")
        grid_rows = _positive_int(banana_cfg.get("grid_rows", 7), "randomization.banana.grid_rows")
        yaw_steps = _positive_int(banana_cfg.get("yaw_steps", 6), "randomization.banana.yaw_steps")
        xs = np.linspace(x_min, x_max, grid_cols)
        ys = np.linspace(y_min, y_max, grid_rows)
        yaws = [2.0 * math.pi * i / yaw_steps for i in range(yaw_steps)]
        return [(float(x), float(y), z_spawn, float(yaw)) for x in xs for y in ys for yaw in yaws]
    if mode == "uniform":
        if "seed" not in banana_cfg or "count" not in banana_cfg:
            raise ValueError("randomization.banana uniform mode requires 'seed' and 'count'")
        seed = banana_cfg["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"randomization.banana.seed must be an integer, got {seed!r}")
        count = _positive_int(banana_cfg["count"], "randomization.banana.count")
        rng = random.Random(seed)
        return [
            (
                rng.uniform(x_min, x_max),
                rng.uniform(y_min, y_max),
                z_spawn,
                rng.uniform(0.0, 2.0 * math.pi),
            )
            for _ in range(count)
        ]
    raise ValueError(f"randomization.banana mode must be 'grid' or 'uniform', got {mode!r}")


def arm_rest_pairs(layout: dict) -> list[tuple[str, float]]:
    """The validated (arm joint name, rest qpos) pairs from the layout.

    The randomisation section's ``arm_joint_names`` / ``arm_rest_qpos`` feed
    both the scene task's rest trajectories and the MuJoCo backend's
    ``ResetWorld`` joint override, so the two arrays are parsed and validated
    once here: both fields must be arrays (a string would otherwise be
    iterated character by character), names must be unique non-empty
    strings, the arrays equally long, every rest value a finite number.
    ``ValueError`` surfaces in the consumer's constructor — before any
    motion is commanded.
    """
    rand = (layout or {}).get("randomization", {}) or {}
    names = rand.get("arm_joint_names")
    qpos = rand.get("arm_rest_qpos")
    names = [] if names is None else names
    qpos = [] if qpos is None else qpos
    for field, value in (("randomization.arm_joint_names", names), ("randomization.arm_rest_qpos", qpos)):
        if not isinstance(value, list | tuple):
            raise ValueError(f"{field} must be a list, got {value!r}")
    if len(names) != len(qpos):
        raise ValueError(
            f"randomization.arm_joint_names ({len(names)}) and arm_rest_qpos ({len(qpos)}) must have the same length"
        )
    pairs = []
    for name, value in zip(names, qpos, strict=True):
        if not isinstance(name, str) or not name:
            raise ValueError(f"randomization.arm_joint_names entries must be non-empty strings, got {name!r}")
        pairs.append((name, _finite_number(value, "randomization.arm_rest_qpos entries")))
    if len({name for name, _ in pairs}) != len(pairs):
        raise ValueError("randomization.arm_joint_names must be unique")
    return pairs


def _quat_from_extrinsic_xyz(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Extrinsic XYZ (URDF rpy) rotation as a MuJoCo w-x-y-z quaternion."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    # q = qz(yaw) @ qy(pitch) @ qx(roll), scalar first
    return (
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    )


def _layout_dynamic_objects(layout: dict) -> list[tuple[str, list[float], list[float]]]:
    """The scene's dynamic objects as (name, position, rotation_euler_rad)."""
    out = []
    for obj in layout.get("objects", []) or []:
        if str(obj.get("type", "")).lower() != "dynamic":
            continue
        out.append(
            (
                str(obj.get("name", "")),
                [float(v) for v in (obj.get("position") or [0.0, 0.0, 0.0])],
                [float(v) for v in (obj.get("rotation_euler_rad") or [0.0, 0.0, 0.0])],
            )
        )
    return out


def scene_dynamic_objects(scene_name: str) -> list[str]:
    """Names of the scene's dynamic objects (``type: dynamic`` in layout.yaml).

    The MuJoCo runtime provider publishes exactly these bodies through the
    FreeJointStatePublisherPlugin, and the scene task's MuJoCo backend
    observes them on the resulting topic. Unknown scenes (or scenes without
    dynamic objects) return an empty list — no plugin is configured for them.
    """
    return [name for name, _position, _rpy in _layout_dynamic_objects(get_scene_layout(scene_name))]


def free_joint_states_topic(scene: str) -> str:
    """The scene-namespaced free-joint observation topic on the MuJoCo platform.

    ``/sim/<scene>/free_joint_states`` mirrors Gazebo's
    ``/sim/<scene>/object_poses``: one scene-namespaced absolute stream with
    every dynamic object's live pose. Both the runtime provider (which
    configures the plugin's ``topic`` parameter) and the scene task's
    ``MujocoBackend`` (which subscribes) resolve the name through this
    function, so the two ends can never drift apart.
    """
    return f"/sim/{scene}/free_joint_states"


def get_scene_file(scene_name: str, platform: str) -> Path:
    """Return the Path to a scene's template file.

    Args:
        scene_name: Scene name (e.g. ``"pick_banana"``, ``"empty"``).
        platform:   Simulator platform — ``"gazebo"`` or ``"mujoco"``.

    Returns:
        Absolute :class:`pathlib.Path` to the template file.

    Raises:
        ValueError:        If *platform* is not ``"gazebo"`` or ``"mujoco"``.
        FileNotFoundError: If the template file does not exist in the
                           installed sim_models share directory.
    """
    ext = _PLATFORM_EXT.get(platform)
    if ext is None:
        raise ValueError(f"Unknown platform: {platform!r}. Supported: {list(_PLATFORM_EXT)}")
    pkg_share = Path(get_package_share_directory("sim_models"))
    path = pkg_share / "scenes" / scene_name / f"{scene_name}{ext}"
    if not path.exists():
        raise FileNotFoundError(
            f"Scene template not found: {path}\n  Did you run 'colcon build --packages-select sim_models'?"
        )
    return path


def get_gazebo_world_path(scene_name: str) -> Path:
    """Generate a Gazebo world file with resolved mesh paths.

    Reads ``{scene_name}.world.template``, replaces ``{{MESHES_DIR}}``
    with the absolute path to the installed ``meshes/`` directory, and
    writes the result to ``/tmp/sim_models_{scene_name}.world``.

    Args:
        scene_name: Scene name (e.g. ``"pick_banana"``).

    Returns:
        :class:`pathlib.Path` to the generated ``.world`` file in ``/tmp/``.
    """
    tmpl = get_scene_file(scene_name, "gazebo")
    meshes_dir = tmpl.parent / "meshes"
    content = tmpl.read_text().replace(_MESHES_PLACEHOLDER, str(meshes_dir))
    out = Path(f"/tmp/sim_models_{scene_name}.world")
    out.write_text(content)
    return out


def composed_model_path(scene: str | None) -> Path:
    """The single composed-model location (D3), keyed by ``ROS_DOMAIN_ID``.

    Resolves to ``<tempdir>/ibrobot_sim_models/<ROS_DOMAIN_ID or 0>/<scene>.mjb``
    (``robot`` when no scene is composed), so concurrent launches on different
    domains (separate worktrees, e2e domains) do not overwrite each other's
    model. The provider (writer) and the scene task (reader) both resolve the
    composed model through this function.
    """
    domain = str(os.environ.get("ROS_DOMAIN_ID", "") or "0")
    name = scene if scene else "robot"
    return Path(tempfile.gettempdir()) / "ibrobot_sim_models" / domain / f"{name}.mjb"


def compose_mujoco_model(robot_mjcf_path: str, scene: str | None, sdf_mesh_names) -> Path:
    """Robot-agnostic robot+scene composition for the MuJoCo provider (D3).

    Substitutes the robot MJCF into the named scene template (``scene=None``
    compiles the robot MJCF alone), enables SDF collision on the given mesh
    names, compiles and saves the ``.mjb`` at ``composed_model_path(scene)``.
    The intermediate XML is kept beside it for debugging. Falls back to the
    ``.xml`` path only when the ``mujoco`` package is missing; the provider
    treats that as fatal. Issue #195: no keyframes are injected any more —
    scene placements travel through ``ResetWorld`` state overrides built
    from the layout, not through compiled-in ``random_KKK`` keys.

    Args:
        robot_mjcf_path:  Absolute path to the fully substituted robot MJCF.
        scene:            Scene name (e.g. ``"pick_banana"``, ``"empty"``),
                          or None for a robot-only model.
        sdf_mesh_names:   Mesh asset names that need SDF non-convex collision
                          (robot-specific input, e.g. the gripper fingers).

    Returns:
        :class:`pathlib.Path` to the compiled ``.mjb`` (or the ``.xml``
        fallback when the ``mujoco`` package is unavailable).
    """
    robot_path = Path(robot_mjcf_path)
    if scene:
        tmpl = get_scene_file(scene, "mujoco")
        meshes_dir = tmpl.parent / "meshes"
        content = tmpl.read_text().replace(_MESHES_PLACEHOLDER, str(meshes_dir))
        content = content.replace(_ROBOT_XML_PLACEHOLDER, str(robot_path))
    else:
        content = robot_path.read_text()

    xml_out = composed_model_path(scene).with_suffix(".xml")
    xml_out.parent.mkdir(parents=True, exist_ok=True)
    xml_out.write_text(content)

    try:
        import mujoco  # noqa: PLC0415
    except ImportError:
        logging.warning("mujoco not available; composed model left as XML at %s", xml_out)
        return xml_out

    try:
        spec = mujoco.MjSpec.from_string(content)
        sdf_names = {str(name) for name in sdf_mesh_names}
        for mesh in spec.meshes:
            if mesh.name in sdf_names:
                mesh.needsdf = True
        model = spec.compile()
        mjb_out = composed_model_path(scene)
        mujoco.mj_saveModel(model, str(mjb_out))
        return mjb_out
    except Exception as exc:  # noqa: BLE001
        logging.warning("MuJoCo compile failed, falling back to XML: %s", exc)
        return xml_out


def get_scene_layout(scene_name: str) -> dict:
    """Return the parsed layout.yaml for a scene, or an empty dict if not found.

    Args:
        scene_name: Scene name (e.g. ``"pick_banana"``).

    Returns:
        Parsed YAML dict.  Keys include ``robot_spawn`` (optional) and ``objects``.
    """
    pkg_share = Path(get_package_share_directory("sim_models"))
    layout_path = pkg_share / "scenes" / scene_name / "layout.yaml"
    if not layout_path.exists():
        return {}
    return yaml.safe_load(layout_path.read_text()) or {}


def get_scene_task(scene_name: str) -> dict | None:
    """Return the scene's ``task:`` declaration, or None when it declares none.

    A scene task declaration names the executable the world owner starts
    after runtime readiness and the physics platforms it supports::

        task:
          executable: pick_banana_task_node
          platforms: [mujoco, gazebo]
    """
    task = get_scene_layout(scene_name).get("task")
    return task if isinstance(task, dict) else None

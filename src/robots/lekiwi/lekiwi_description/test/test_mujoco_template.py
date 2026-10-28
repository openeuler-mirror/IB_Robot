"""The LeKiwi MuJoCo robot template (lekiwi-mujoco-runtime-platform D2).

Structural pins on ``mujoco/lekiwi_base.xml.template`` (the runtime injects
peripherals into it, so the template itself must stay peripheral-free), the
D2b decimated visual meshes, and the pinned physics behaviour: the composed
model settles on frictionless wheel contacts — every active wheel–ground
contact has ``dim == 1`` — through the real ``compose_mujoco_model`` path.
"""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_PACKAGE = Path(__file__).resolve().parents[1]
_TEMPLATE = _PACKAGE / "mujoco" / "lekiwi_base.xml.template"
_MESHES = _PACKAGE / "meshes"
_MUJOCO_MESHES = _MESHES / "mujoco"

#: MuJoCo 3.12.0's STL decoder face limit (why meshes/mujoco/ exists, D2b).
MUJOCO_STL_FACE_LIMIT = 200000


def _stl_face_count(path: Path) -> int:
    with open(path, "rb") as handle:
        return struct.unpack("<I", handle.read(84)[80:84])[0]


def test_template_declares_the_mujoco_conventions_and_stays_peripheral_free():
    root = ET.parse(_TEMPLATE).getroot()
    assert root.tag == "mujoco"

    compiler = root.find("compiler")
    assert compiler.get("angle") == "radian"
    assert compiler.get("eulerseq") == "XYZ"  # extrinsic XYZ == URDF rpy
    assert compiler.get("meshdir") == "{{MESHES_DIR}}"

    base = root.find("worldbody/body[@name='base_link']")
    assert base.get("pos") == "{{ROBOT_BASE_POS}}"
    assert base.get("quat") == "{{ROBOT_BASE_QUAT}}"
    freejoint = base.find("freejoint")
    assert freejoint is not None and freejoint.get("name") == "floating_base_joint"
    inertial = base.find("inertial")
    # the URDF base inertial, unchanged
    assert inertial.get("mass") == "2.0"
    assert inertial.get("diaginertia") == "0.01 0.01 0.02"
    # the chassis collision hull clears the ground (the wheels reach lower);
    # group 3 keeps it out of the default render (review round 2) without
    # touching its collision role
    chassis = base.find("geom[@name='chassis']")
    assert chassis.get("type") == "box" and chassis.get("pos") == "0 0 0.05"
    assert chassis.get("group") == "3"
    assert float(chassis.get("pos").split()[2]) - float(chassis.get("size").split()[2]) > -0.0406

    # D2a: wheel joint/actuator names (a PidROS topic token may not start
    # with a digit); D2: frictionless spheres outranking the scene planes
    # (the wheel collision parameters live in the wheel_collision default
    # class — one source for all three geoms)
    wheel_default = root.find("default/default[@class='wheel_collision']/geom")
    assert wheel_default.get("type") == "sphere" and wheel_default.get("size") == "0.051"
    assert wheel_default.get("condim") == "1" and wheel_default.get("priority") == "2"
    assert wheel_default.get("friction") == "0 0 0"
    # the collision spheres render hidden too (group 3, above the viewer's
    # default-visible groups 0-2); contacts and rays ignore the group
    assert wheel_default.get("group") == "3"
    wheels = {
        body.get("name"): body for body in root.findall("worldbody/body/body") if body.get("name", "").endswith("wheel")
    }
    assert set(wheels) == {"left_wheel", "back_wheel", "right_wheel"}
    for wheel_name, body in wheels.items():
        index = {"left_wheel": "7", "back_wheel": "8", "right_wheel": "9"}[wheel_name]
        joint = body.find("joint")
        assert joint.get("name") == f"wheel_{index}"
        assert joint.get("type") == "hinge" and joint.get("axis") == "0 0 -1"
        assert body.find("geom[@class='wheel_collision']") is not None
        visual = body.find("geom[@class='visual']")
        assert visual.get("type") == "mesh" and visual.get("mesh") == "omni_wheel"
    actuators = {actuator.get("name"): actuator for actuator in root.findall("actuator/velocity")}
    assert set(actuators) == {"wheel_7", "wheel_8", "wheel_9"}
    for name, actuator in actuators.items():
        assert actuator.get("joint") == name
        assert actuator.get("kv") == "0.6"
        # 1.2x the profile's max_wheel_radps 4.602
        assert actuator.get("ctrlrange") == "-5.5224 5.5224"

    # peripheral-free: the runtime injects sites, sensors and the engine
    # extension from the inventory (D5.2); the template ships none of them
    assert root.find("sensor") is None
    assert root.find("extension") is None
    assert root.find(".//site") is None
    assert root.find(".//camera") is None

    # visual geoms never collide (D2/D2b): rays still hit them (alpha != 0),
    # but they generate no contacts (the parameters live in the visual
    # default class every visual geom references)
    visual_default = root.find("default/default[@class='visual']/geom")
    assert visual_default.get("contype") == "0" and visual_default.get("conaffinity") == "0"
    assert visual_default.get("group") == "1"
    visual_geoms = [geom for geom in root.iter("geom") if geom.get("class") == "visual"]
    assert len(visual_geoms) == 4  # the chassis plate + three wheels
    for geom in visual_geoms:
        assert geom.get("contype") is None and geom.get("conaffinity") is None


def test_visual_geoms_reference_the_decimated_meshes_only():
    root = ET.parse(_TEMPLATE).getroot()
    meshes = {mesh.get("name"): mesh.get("file") for mesh in root.findall("asset/mesh")}
    assert meshes == {"lekiwi_base": "lekiwi_base.stl", "omni_wheel": "omni_wheel.stl"}


@pytest.mark.parametrize("name", ["lekiwi_base.stl", "omni_wheel.stl"])
def test_decimated_visual_meshes_exist_under_the_face_limit(name):
    """D2b: the decimated copies load in MuJoCo (face count within the limit)."""
    path = _MUJOCO_MESHES / name
    assert path.is_file(), f"missing decimated visual mesh {path}"
    assert _stl_face_count(path) <= MUJOCO_STL_FACE_LIMIT
    assert _stl_face_count(_MESHES / name) > MUJOCO_STL_FACE_LIMIT  # why the copies exist


def test_composed_model_settles_on_frictionless_wheel_contacts(tmp_path, monkeypatch):
    """The pinned D2 physics test: compose through the real path, step, and
    every active wheel–ground contact has ``dim == 1`` (the frictionless
    sphere's priority-2 parameters govern every wheel–plane pair)."""
    mujoco = pytest.importorskip("mujoco")
    from sim_models.scene_compiler import compose_mujoco_model

    monkeypatch.setenv("ROS_DOMAIN_ID", "lekiwi_description_mujoco_template_test")
    content = (
        _TEMPLATE.read_text(encoding="utf-8")
        .replace("{{MESHES_DIR}}", str(_MUJOCO_MESHES))
        .replace("{{ROBOT_BASE_POS}}", "0 0 0.05")
        .replace("{{ROBOT_BASE_QUAT}}", "1 0 0 0")
    )
    robot = tmp_path / "robot.xml"
    robot.write_text(content, encoding="utf-8")
    composed = compose_mujoco_model(str(robot), "empty", ())
    # no peripherals -> no engine extension -> the .mjb artifact
    assert composed.suffix == ".mjb" and composed.exists()
    model = mujoco.MjModel.from_binary_path(str(composed))
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, nstep=1000)
    mujoco.mj_forward(model, data)

    wheels = {"wheel_7", "wheel_8", "wheel_9"}
    contacts = []
    for i in range(data.ncon):
        contact = data.contact[i]
        first = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        second = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        contacts.append((first, second, int(contact.dim)))
    # settled on the invisible support plane: exactly one contact per wheel
    assert len(contacts) == 3, contacts
    for first, second, dim in contacts:
        pair = {first, second}
        assert len(pair & wheels) == 1, contacts
        assert dim == 1, contacts


@pytest.mark.parametrize("wheel_geom", ["wheel_7", "wheel_8", "wheel_9"])
def test_rays_still_hit_the_group3_collision_spheres(wheel_geom, monkeypatch, tmp_path):
    """Review round 2: the collision geoms moved to rendering group 3 — the
    lidar ray cast has no geom-group filter, so the wheel spheres stay in the
    self-hit surface. Pinned directly: a vertical ray above each wheel with a
    geomgroup filter that enables ONLY group 3 still hits, and the hit geom is
    that wheel's collision sphere (with the filter the lidar actually uses —
    NULL, every geom, group 3 included — the hit is a fortiori there)."""
    mujoco = pytest.importorskip("mujoco")
    import numpy as np

    from sim_models.scene_compiler import compose_mujoco_model

    monkeypatch.setenv("ROS_DOMAIN_ID", "lekiwi_description_mujoco_template_test")
    content = (
        _TEMPLATE.read_text(encoding="utf-8")
        .replace("{{MESHES_DIR}}", str(_MUJOCO_MESHES))
        .replace("{{ROBOT_BASE_POS}}", "0 0 0.05")
        .replace("{{ROBOT_BASE_QUAT}}", "1 0 0 0")
    )
    robot = tmp_path / "robot.xml"
    robot.write_text(content, encoding="utf-8")
    composed = compose_mujoco_model(str(robot), "empty", ())
    model = mujoco.MjModel.from_binary_path(str(composed))
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, nstep=1000)
    mujoco.mj_forward(model, data)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, wheel_geom)
    assert geom_id >= 0 and model.geom_group[geom_id] == 3
    body_id = model.geom_bodyid[geom_id]
    wheel_x, wheel_y, wheel_z = data.xpos[body_id]
    base_x, base_y, _ = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")]
    # A horizontal ray from outside the chassis along the wheel's azimuth,
    # at the sphere's centre height: the first group-3 surface it meets is
    # the sphere's outer side (the chassis hull stays inside radial 0.13 m,
    # the sphere's outer surface sits at 0.183 m — nothing occludes it).
    outward = np.array([wheel_x - base_x, wheel_y - base_y])
    outward /= np.linalg.norm(outward)
    only_group3 = np.zeros(mujoco.mjNGROUP, dtype=np.uint8)
    only_group3[3] = 1
    hit_geom = np.zeros(1, dtype=np.int32)
    distance = mujoco.mj_ray(
        model,
        data,
        np.array([base_x + 0.5 * outward[0], base_y + 0.5 * outward[1], wheel_z]),
        np.array([-outward[0], -outward[1], 0.0]),
        only_group3,
        True,
        -1,
        hit_geom,
    )
    assert 0 < distance < 0.5, f"no group-3 ray hit towards {wheel_geom}: {distance}"
    assert int(hit_geom[0]) == geom_id, (
        f"first group-3 hit towards {wheel_geom} is geom {hit_geom[0]}, "
        f"expected the collision sphere (distance {distance:.4f})"
    )

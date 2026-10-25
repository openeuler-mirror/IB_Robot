"""The nav_room scene (lekiwi-gazebo-runtime-platform D8, D14.5, D14.7).

The room-level navigation scene: the compiler resolves its world template
and a real Gazebo server loads the resolved world headlessly. The scene's
static collision geometry is data (``layout.yaml`` ``static_geometry``,
D14.7): every simulator template of the scene must carry exactly that
geometry, and the gate route (D14.5) must keep its swept footprint and the
collision monitor's look-ahead clear of it.
"""

from __future__ import annotations

import math
import os
import subprocess
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from sim_models.scene_compiler import get_gazebo_world_path, get_scene_layout

_SCENE_DIR = Path(__file__).resolve().parents[1] / "scenes" / "nav_room"

# The Nav2 footprint disc radius (robot_radius, robot_navigation
# nav2_params_sim.yaml local_costmap) and the collision monitor's approach
# look-ahead (time_before_collision, same file's collision_monitor).
FOOTPRINT_RADIUS = 0.22
TIME_BEFORE_COLLISION = 1.5
#: Clearance rule margins (D14.5): the swept footprint stays at least this
#: far from every wall/obstacle, and the free distance ahead at a segment
#: end is at least speed x look-ahead plus this much.
SIDE_MARGIN = 0.05
AHEAD_MARGIN = 0.05

#: The D14.5 gate route, read from the SAME layout data the gate executes
#: (one definition, no drift): west half of the room, 7.0 m of translation,
#: two in-place turns, two lateral segments, closed. Translation speeds are
#: signed (segment 6 drives body -y); the clearance rule uses the magnitude.
ROUTE = tuple(get_scene_layout("nav_room")["gate_route"])


def _gazebo_available() -> bool:
    try:
        return subprocess.run(["ign", "gazebo", "--version"], capture_output=True, timeout=10).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def test_nav_room_layout_declares_the_spawn():
    layout = get_scene_layout("nav_room")
    assert layout["scene_name"] == "nav_room"
    # D8: the spawn pose of the navigation scene (z lifts the wheels)
    assert layout["robot_spawn"] == {"x": -1.5, "y": -1.5, "z": 0.05, "yaw": 0.0}


def test_nav_room_world_resolves_with_the_room_geometry():
    world = get_gazebo_world_path("nav_room")
    assert world.is_file()
    content = world.read_text(encoding="utf-8")
    # same geometry as nav_test.world, so sim_map.yaml stays valid
    for model in ("wall_north", "wall_south", "wall_east", "wall_west", "obstacle_box", "obstacle_cylinder"):
        assert f'name="{model}"' in content, model
    assert '<world name="demo">' in content
    # the systems runtime sensors need
    for system in ("Physics", "UserCommands", "SceneBroadcaster", "Sensors", "Imu"):
        assert f"gz::sim::systems::{system}" in content, system


@pytest.mark.skipif(not _gazebo_available(), reason="Gazebo (ign gazebo) is not available")
def test_nav_room_world_loads_in_gazebo():
    world = get_gazebo_world_path("nav_room")
    environment = dict(os.environ)
    # both partition variables: Fortress (ign, gz-transport11) reads only
    # IGN_PARTITION, newer gz reads GZ_PARTITION — set both so this load
    # run stays on its own gz bus on either stack
    environment["GZ_PARTITION"] = f"nav_room_load_{uuid.uuid4().hex[:8]}"
    environment["IGN_PARTITION"] = environment["GZ_PARTITION"]
    result = subprocess.run(
        ["ign", "gazebo", "-s", "-r", "--iterations", "10", str(world)],
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
    )
    assert result.returncode == 0, f"gazebo failed to load {world}:\n{result.stdout}\n{result.stderr}"


def test_nav_room_layout_is_loadable_yaml_roundtrip():
    layout = get_scene_layout("nav_room")
    assert yaml.safe_load(yaml.safe_dump(layout)) == layout


# ---------------------------------------------------------------------------
# D14.7: scene geometry is backend-neutral data
# ---------------------------------------------------------------------------


def _parse_pose(text: str) -> list[float]:
    values = [float(value) for value in (text or "").split()]
    assert len(values) == 6, f"a pose needs 6 values, got {text!r}"
    return values


def _template_geometry(template: Path) -> dict[str, dict]:
    """The static collision geometry of one scene template, by model name.

    Only box/cylinder collisions are recorded — static models outside the
    layout's ``static_geometry`` (the ground plane) carry other primitives
    and are not part of the compared surface (D14.7: static collision
    geometry means the walls and obstacles only).
    """
    geometry: dict[str, dict] = {}
    root = ET.parse(template).getroot()
    for model in root.iter("model"):
        name = model.get("name")
        static = model.find("static")
        if name is None or static is None or (static.text or "").strip().lower() != "true":
            continue  # dynamic models and non-<static>true</static> spellings are not compared
        collision = model.find("./link/collision/geometry")
        assert collision is not None, f"{template.name}: model {name!r} has no collision geometry"
        box = collision.find("box")
        cylinder = collision.find("cylinder")
        if box is None and cylinder is None:
            continue  # primitives outside the convention (the ground plane)
        pose = model.find("pose")
        entry = {"pose": _parse_pose(pose.text if pose is not None else "0 0 0 0 0 0")}
        if box is not None:
            entry["primitive"] = "box"
            entry["size"] = [float(value) for value in (box.find("size").text or "").split()]
        else:
            entry["primitive"] = "cylinder"
            entry["size"] = [float(cylinder.find("radius").text), float(cylinder.find("length").text)]
        geometry[name] = entry
    return geometry


def _assert_template_matches_static_geometry(template: Path, static_geometry: list[dict]) -> None:
    """Every layout element appears in the template with the same geometry.

    Fails naming the element and both values (layout vs template, D14.7), so
    template drift cannot silently move a wall the route was cleared
    against.
    """
    rendered = _template_geometry(template)
    # both directions (D14.7): every layout element must appear in the
    # template, and the template must not carry EXTRA box/cylinder
    # collision geometry the clearance rule has never seen
    extras = sorted(set(rendered) - {element["name"] for element in static_geometry})
    assert not extras, (
        f"{template.name}: static box/cylinder collision geometry absent from the layout static_geometry "
        f"(extra elements: {extras}) — every wall/obstacle the route clearance runs against must be listed"
    )
    for element in static_geometry:
        name = element["name"]
        assert name in rendered, f"{template.name}: static element {name!r} is absent from the template"
        got = rendered[name]
        assert got["primitive"] == element["primitive"], (
            f"{template.name}: element {name!r} is a {got['primitive']} in the template but a "
            f"{element['primitive']} in the layout"
        )
        assert got["size"] == pytest.approx(element["size"]), (
            f"{template.name}: element {name!r} size is {got['size']} in the template "
            f"but {element['size']} in the layout"
        )
        assert got["pose"] == pytest.approx(element["pose"]), (
            f"{template.name}: element {name!r} pose is {got['pose']} in the template "
            f"but {element['pose']} in the layout"
        )


def test_every_scene_template_carries_the_layout_static_geometry():
    layout = get_scene_layout("nav_room")
    static_geometry = layout["static_geometry"]
    assert static_geometry, "nav_room must declare its static collision geometry (D14.7)"
    templates = sorted(_SCENE_DIR.glob("nav_room.*.template"))
    assert templates, f"no nav_room scene templates under {_SCENE_DIR}"
    for template in templates:
        _assert_template_matches_static_geometry(template, static_geometry)


def test_a_shifted_obstacle_copy_fails_the_template_consistency(tmp_path):
    """D14.7 template drift: a temporary copy with a moved obstacle fails,
    naming the element and both values (an in-test fixture, not a shipped
    file)."""
    layout = get_scene_layout("nav_room")
    static_geometry = layout["static_geometry"]
    original = (_SCENE_DIR / "nav_room.world.template").read_text(encoding="utf-8")
    shifted = original.replace("<pose>0 0 0.25 0 0 0</pose>", "<pose>0.9 0 0.25 0 0 0</pose>", 1)
    assert shifted != original, "the fixture must actually shift an obstacle"
    shifted_template = tmp_path / "nav_room.world.template"
    shifted_template.write_text(shifted, encoding="utf-8")
    with pytest.raises(AssertionError, match=r"obstacle_box.*template.*layout") as excinfo:
        _assert_template_matches_static_geometry(shifted_template, static_geometry)
    assert "0.9" in str(excinfo.value), "the failure message must carry both values"


# ---------------------------------------------------------------------------
# D14.5: the gate route keeps its footprint and look-ahead clear
# ---------------------------------------------------------------------------


def _point_segment_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    (px, py), (ax, ay), (bx, by) = point, start, end
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _point_box_distance(point, center, half) -> float:
    dx = max(abs(point[0] - center[0]) - half[0], 0.0)
    dy = max(abs(point[1] - center[1]) - half[1], 0.0)
    return math.hypot(dx, dy)


def _segment_crosses_box(start, end, center, half) -> bool:
    """Slab test: does the segment [start, end] reach the box's interior?"""
    delta = (end[0] - start[0], end[1] - start[1])
    t_min, t_max = 0.0, 1.0
    for axis in (0, 1):
        if abs(delta[axis]) < 1e-12:
            if abs(start[axis] - center[axis]) > half[axis]:
                return False
            continue
        t1 = (center[axis] - half[axis] - start[axis]) / delta[axis]
        t2 = (center[axis] + half[axis] - start[axis]) / delta[axis]
        t_min = max(t_min, min(t1, t2))
        t_max = min(t_max, max(t1, t2))
        if t_min > t_max:
            return False
    return True


class _Obstacle:
    """One planar static element of the scene geometry (D14.7)."""

    def __init__(self, element: dict):
        self.name = element["name"]
        pose = element["pose"]
        roll, pitch, yaw = pose[3], pose[4], pose[5]
        assert roll == 0.0 and pitch == 0.0 and yaw == 0.0, (
            f"{self.name}: the planar clearance check needs axis-aligned scene geometry, got rpy {pose[3:]}"
        )
        self.center = (pose[0], pose[1])
        if element["primitive"] == "box":
            self.kind = "box"
            self.half = (element["size"][0] / 2.0, element["size"][1] / 2.0)
            self.corners = [
                (self.center[0] + sx * self.half[0], self.center[1] + sy * self.half[1])
                for sx in (-1, 1)
                for sy in (-1, 1)
            ]
        else:
            self.kind = "circle"
            self.radius = element["size"][0]

    def _contains(self, point) -> bool:
        if self.kind == "circle":
            return math.hypot(point[0] - self.center[0], point[1] - self.center[1]) <= self.radius
        return (
            abs(point[0] - self.center[0]) <= self.half[0] + 1e-12
            and abs(point[1] - self.center[1]) <= self.half[1] + 1e-12
        )

    def distance_to_segment(self, start, end) -> float:
        """Exact planar distance between the segment and the obstacle region."""
        if self.kind == "circle":
            return max(0.0, _point_segment_distance(self.center, start, end) - self.radius)
        if self._contains(start) or self._contains(end) or _segment_crosses_box(start, end, self.center, self.half):
            return 0.0
        # disjoint: the closest pair is an endpoint against the box (clamp)
        # or a box corner against the segment — the parallel sliding case
        # is covered by the corners projecting onto the segment interior
        return min(
            _point_box_distance(start, self.center, self.half),
            _point_box_distance(end, self.center, self.half),
            *(_point_segment_distance(corner, start, end) for corner in self.corners),
        )

    def ray_distance(self, origin, direction, expand: float) -> float | None:
        """Travel distance from ``origin`` along the (infinite) ray before the
        robot's (expanded) footprint touches this obstacle; None on a miss."""
        if self.kind == "circle":
            radius = self.radius + expand
            ox, oy = origin[0] - self.center[0], origin[1] - self.center[1]
            b = ox * direction[0] + oy * direction[1]
            c = ox * ox + oy * oy - radius * radius
            if c <= 0.0:
                return 0.0  # the footprint already overlaps the obstacle
            disc = b * b - c
            if disc < 0.0 or b >= 0.0:
                return None
            return -b - math.sqrt(disc)
        half = (self.half[0] + expand, self.half[1] + expand)
        # ray slab test: the entry parameter along the direction, t >= 0
        t_min, t_max = 0.0, math.inf
        for axis in (0, 1):
            if abs(direction[axis]) < 1e-12:
                if abs(origin[axis] - self.center[axis]) > half[axis]:
                    return None
                continue
            t1 = (self.center[axis] - half[axis] - origin[axis]) / direction[axis]
            t2 = (self.center[axis] + half[axis] - origin[axis]) / direction[axis]
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
        if t_max < t_min:
            return None
        return t_min


def _route_clearance_violations(route, obstacles):
    """(segment, element, margin, rule) for every clearance-rule breach."""
    violations = []
    for entry in route:
        segment = entry["segment"]
        if entry["kind"] == "rotate":
            point = entry["point"]
            for obstacle in obstacles:
                margin = obstacle.distance_to_segment(point, point) - FOOTPRINT_RADIUS
                if margin < SIDE_MARGIN:
                    violations.append((segment, obstacle.name, margin, "swept footprint"))
            continue
        start, end = entry["start"], entry["end"]
        for obstacle in obstacles:
            margin = obstacle.distance_to_segment(start, end) - FOOTPRINT_RADIUS
            if margin < SIDE_MARGIN:
                violations.append((segment, obstacle.name, margin, "swept footprint"))
        # the collision monitor's approach polygon at the segment end: the
        # free distance ahead must cover speed x look-ahead plus the margin
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        direction = ((end[0] - start[0]) / length, (end[1] - start[1]) / length)
        required = abs(entry["speed"]) * TIME_BEFORE_COLLISION + AHEAD_MARGIN
        hits = [(obstacle.ray_distance(end, direction, FOOTPRINT_RADIUS), obstacle) for obstacle in obstacles]
        hits = [(distance, obstacle) for distance, obstacle in hits if distance is not None]
        if hits:
            free, nearest = min(hits, key=lambda item: item[0])
            if free < required:
                violations.append((segment, nearest.name, free - required, "look-ahead"))
    return violations


def test_the_gate_route_clears_the_scene_geometry():
    obstacles = [_Obstacle(element) for element in get_scene_layout("nav_room")["static_geometry"]]
    violations = _route_clearance_violations(ROUTE, obstacles)
    assert not violations, "the D14.5 route must stay clear of the scene geometry: " + "; ".join(
        f"segment {segment} violates the {rule} rule against {element} by {margin:.3f} m"
        for segment, element, margin, rule in violations
    )


def test_a_shifted_obstacle_fails_the_route_clearance():
    """D14.7/D14.5: moving an obstacle into the route corridor must fail the
    clearance check, naming the segment, the element and the margin (an
    in-test fixture, not a shipped file)."""
    static_geometry = yaml.safe_load(yaml.safe_dump(get_scene_layout("nav_room")["static_geometry"]))
    box = next(element for element in static_geometry if element["name"] == "obstacle_box")
    # onto segment 1's corridor: the swept footprint then overlaps the box
    box["pose"] = [-1.1, -1.5, 0.25, 0.0, 0.0, 0.0]
    obstacles = [_Obstacle(element) for element in static_geometry]
    violations = _route_clearance_violations(ROUTE, obstacles)
    assert violations, "a shifted obstacle on the route must violate the clearance rule"
    segment, element, margin, rule = violations[0]
    assert segment == 1 and element == "obstacle_box" and rule == "swept footprint"
    assert margin < 0.0

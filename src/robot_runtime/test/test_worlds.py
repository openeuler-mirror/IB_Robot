"""Physics worlds provide the systems runtime sensors need (D5).

Every world a provider may run in on a physics platform — the provider's
standalone minimal world and every shipped simulation scene template (the
generic fallback world is the sim_models ``empty`` scene, one of those
templates) — must load the simulator systems the sensor kinds the runtime
core can generate require (cameras and lidars render through the sensors
system; IMU sensors need the IMU system), so a declared sensor publishes
without world-owner knowledge of the robot.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]

# (path, label) of every world the runtime may run in: the provider's
# standalone minimal world and every shipped scene template — the generic
# fallback world is the sim_models empty scene, already covered by the scene
# glob (task 4.1's nav_room joins the scene glob when it lands).
_WORLDS = [
    (_ROOT / "src/robot_runtime/worlds/empty.sdf", "provider standalone world"),
    *(
        (path, f"shipped scene template {path.parent.name}")
        for path in sorted((_ROOT / "src/sim_models/scenes").glob("*/*.world.template"))
    ),
]

_REQUIRED_PLUGINS = {
    "physics": "gz::sim::systems::Physics",
    "user commands": "gz::sim::systems::UserCommands",
    "scene broadcaster": "gz::sim::systems::SceneBroadcaster",
    "sensors (ogre2)": "gz::sim::systems::Sensors",
    "imu": "gz::sim::systems::Imu",
}


def test_every_runtime_world_loads_the_systems_runtime_sensors_need():
    assert _WORLDS, "no worlds found to check"
    for path, label in _WORLDS:
        assert path.is_file(), f"{label} missing: {path}"
        root = ET.parse(path).getroot()
        worlds = root.findall("world")
        assert worlds, f"{label} has no <world>: {path}"
        for world in worlds:
            names = {plugin.get("name") for plugin in world.findall("plugin")}
            for requirement, expected in _REQUIRED_PLUGINS.items():
                assert expected in names, f"{label} {world.get('name')!r} lacks the {requirement} system ({expected})"
            sensors = [plugin for plugin in world.findall("plugin") if plugin.get("name").endswith("Sensors")]
            for sensor_system in sensors:
                engine = sensor_system.find("render_engine")
                assert engine is not None and (engine.text or "").strip() == "ogre2", (
                    f"{label} {world.get('name')!r} sensors system must render with ogre2"
                )

"""Export a declared topology from YAML, not an inventory of started nodes.

Only sibling base_config inheritance is resolved. Runtime validation, model
availability checks and launch overlays belong to the business loader/launch.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from ibrobot_tracing.topology import bind_robot_topology
from robot_config.loader import load_robot_section


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot_config", type=Path)
    parser.add_argument("--control-mode")
    args = parser.parse_args(argv)
    try:
        config_path, config = load_robot_section(args.robot_config)
        config["_config_path"] = str(config_path)
        topology = bind_robot_topology(config, control_mode=args.control_mode)
        topology.metadata.update(source="robot_config_yaml", provenance="declared", runtime_verified=False)
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({**topology.to_dict(), "schema_version": 2}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

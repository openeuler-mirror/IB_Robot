#!/usr/bin/env python3
"""Migrate legacy SO-101 calibration data to the current schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from so101_hardware.calibration.transfer import (
    DEFAULT_OUTPUT_PREFIX,
    migrate_calibration_file,
)


def _parse_arm_pairs(arms_name_list: str) -> list[tuple[str, str]]:
    parsed = json.loads(arms_name_list)
    return [(str(new_name), str(old_name)) for new_name, old_name in parsed]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transfer legacy SO-101 calibration ranges",
    )
    parser.add_argument(
        "--arms_name_list",
        dest="arms_name_list",
        required=True,
        help="JSON list like [['new', 'old']]",
    )
    parser.add_argument(
        "--new_dir_path",
        dest="new_dir_path",
        required=True,
        help="Directory containing template calibration JSON files",
    )
    parser.add_argument(
        "--old_dir_path",
        dest="old_dir_path",
        required=True,
        help="Directory containing legacy calibration JSON files",
    )
    return parser


def main(args: list[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args=args)
    arm_pairs = _parse_arm_pairs(parsed.arms_name_list)

    new_dir = Path(parsed.new_dir_path).expanduser()
    old_dir = Path(parsed.old_dir_path).expanduser()

    for new_name, old_name in arm_pairs:
        template_path = new_dir / f"{new_name}.json"
        legacy_path = old_dir / f"{old_name}.json"
        output_path = new_dir / f"{DEFAULT_OUTPUT_PREFIX}{new_name}.json"
        migrate_calibration_file(template_path, legacy_path, output_path)
        print(f"Migrated calibration written to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

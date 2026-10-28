from __future__ import annotations

import os
import unittest
from importlib.metadata import distribution
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory

EXPECTED_ENTRY_POINTS = {
    "pipeline_policy_node": "inference_service.pipeline_policy_node:main",
    "pure_inference_node": "inference_service.pure_inference_node:main",
}


class TestInstalledEntryPoints(unittest.TestCase):
    def test_distribution_and_executables_are_current(self) -> None:
        prefix = Path(get_package_prefix("inference_service")).resolve()
        expected_prefix = os.environ.get("INFERENCE_SMOKE_EXPECTED_PREFIX")
        if expected_prefix:
            self.assertEqual(Path(expected_prefix).resolve(), prefix)

        installed = {
            entry.name: entry.value
            for entry in distribution("inference_service").entry_points
            if entry.group == "console_scripts"
        }
        self.assertEqual(EXPECTED_ENTRY_POINTS, installed)

        executable_dir = prefix / "lib" / "inference_service"
        executables = {path.name for path in executable_dir.iterdir() if path.is_file() and os.access(path, os.X_OK)}
        self.assertEqual(set(EXPECTED_ENTRY_POINTS), executables)

    def test_launch_files_are_installed(self) -> None:
        launch_dir = Path(get_package_share_directory("inference_service")) / "launch"
        self.assertTrue((launch_dir / "eval_inference.launch.py").is_file())
        self.assertTrue((launch_dir / "cloud_inference.launch.py").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)

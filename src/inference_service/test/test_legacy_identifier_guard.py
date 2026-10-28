from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def test_active_repository_rejects_removed_inference_identifiers():
    repository_root = Path(__file__).resolve().parents[3]
    # The repository worktree may contain unrelated user documents and
    # experiments. The inference package guard must inspect the committed
    # inference surface, not arbitrary untracked workspace material.
    isolated_root = repository_root / "build" / "inference_legacy_guard"
    if isolated_root.exists():
        shutil.rmtree(isolated_root)
    (isolated_root / "src").mkdir(parents=True)
    for package in ("inference_service", "perception_service", "voice_tts_service", "voice_asr_service"):
        shutil.copytree(repository_root / "src" / package, isolated_root / "src" / package)
    result = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts/check_inference_legacy_identifiers.py"),
            "--root",
            str(isolated_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    shutil.rmtree(isolated_root)

    assert result.returncode == 0, result.stdout + result.stderr


def test_guard_checks_active_markdown_and_ignores_generated_archives(tmp_path):
    repository_root = Path(__file__).resolve().parents[3]
    script = repository_root / "scripts/check_inference_legacy_identifiers.py"
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("Use execution_mode:=distributed\n", encoding="utf-8")
    (tmp_path / "docs" / "migration").mkdir()
    (tmp_path / "docs" / "migration" / "archive.md").write_text("lerobot_policy_node\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "docs/guide.md:1: legacy_launch_argument" in result.stdout
    assert "docs/migration/archive.md" not in result.stdout

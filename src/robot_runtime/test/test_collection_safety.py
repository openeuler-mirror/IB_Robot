"""Unit tests of the shared collection-safety helper (D10/D15)."""

from __future__ import annotations

from robot_runtime.testing.collection_safety import import_module_with_blocked_imports


def _write_module(directory, name: str, body: str) -> None:
    (directory / f"{name}.py").write_text(body, encoding="utf-8")


def test_a_module_without_the_blocked_imports_collects(tmp_path):
    _write_module(tmp_path, "plain_module", "VALUE = 1\n")
    completed = import_module_with_blocked_imports(
        "plain_module", blocked=("mujoco", "mujoco_ros2_control"), cwd=tmp_path
    )
    assert completed.returncode == 0, completed.stderr


def test_a_module_level_blocked_import_fails_collection(tmp_path):
    _write_module(tmp_path, "eager_module", "import mujoco\nVALUE = 1\n")
    completed = import_module_with_blocked_imports("eager_module", blocked=("mujoco",), cwd=tmp_path)
    assert completed.returncode != 0
    # sys.modules[<name>] = None makes the import raise ModuleNotFoundError
    # (an ImportError subclass) naming the blocked module
    assert "ModuleNotFoundError" in completed.stderr and "mujoco" in completed.stderr


def test_a_lazy_blocked_import_still_fails_only_when_taken(tmp_path):
    # the import inside the function never runs on collection: the module
    # still imports, proving the block poisons without false positives
    _write_module(tmp_path, "lazy_module", "def use():\n    import mujoco\n    return mujoco\n")
    completed = import_module_with_blocked_imports("lazy_module", blocked=("mujoco",), cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr

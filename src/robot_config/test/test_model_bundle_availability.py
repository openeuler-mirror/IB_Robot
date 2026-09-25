"""Every model bundle a robot config references must be fetchable.

This is the half of the model-bundle contract that needs no downloaded assets.
The suites that actually load those bundles are marked ``model_bundle`` and skip
on a fresh clone; this test is what keeps that skip honest. Without it, a typo in
a ``bundle_path`` or a bundle nobody can download would silently turn into a skip
on every machine instead of a failure.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
ROBOTS_DIR = ROOT / "src/robot_config/config/robots"
DOWNLOADER = ROOT / "scripts/download_models.py"


def _allowlisted_bundles() -> set[str]:
    """Read the downloader's allowlist statically.

    Parsed rather than imported: download_models.py pulls in optional runtime
    dependencies, and this check must work on a machine that has downloaded
    nothing. BundleSource's third positional field is the models/ directory name.
    """
    tree = ast.parse(DOWNLOADER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "MODEL_BUNDLE_ALLOWLIST" not in targets:
            continue
        names = set()
        for element in getattr(node.value, "elts", []):
            if not isinstance(element, ast.Call):
                continue
            if len(element.args) >= 3 and isinstance(element.args[2], ast.Constant):
                names.add(element.args[2].value)
            for keyword in element.keywords:
                if keyword.arg == "directory" and isinstance(keyword.value, ast.Constant):
                    names.add(keyword.value.value)
        return names
    raise AssertionError("MODEL_BUNDLE_ALLOWLIST not found in scripts/download_models.py")


def _referenced_bundles() -> dict[str, set[str]]:
    """Map each models/ bundle name to the robot configs that reference it."""
    referenced: dict[str, set[str]] = {}

    def walk(node, origin: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "bundle_path" and isinstance(value, str):
                    parts = Path(value).parts
                    if "models" in parts:
                        name = parts[parts.index("models") + 1]
                        referenced.setdefault(name, set()).add(origin)
                else:
                    walk(value, origin)
        elif isinstance(node, list):
            for item in node:
                walk(item, origin)

    for config in sorted(ROBOTS_DIR.glob("*.yaml")):
        walk(yaml.safe_load(config.read_text(encoding="utf-8")), config.name)
    return referenced


def test_downloader_allowlist_is_loadable():
    assert _allowlisted_bundles(), "the model bundle allowlist is empty"


#: Bundles supplied outside scripts/download_models.py. They are reached through
#: a nested path (models/voice_asr/<model>) or provisioned by another route, so
#: the downloader's allowlist is not the right authority for them. Listing them
#: explicitly keeps the check below meaningful instead of simply weakening it.
EXTERNALLY_PROVISIONED = frozenset({"voice_asr", "grounded_sam2_swint_ogc"})


def test_every_referenced_bundle_is_downloadable():
    """A config may only name a bundle some documented route can provide.

    Otherwise a ``model_bundle`` marker turns into a permanent skip: no machine
    could ever obtain the asset, and the suite guarding it would never run.
    """
    allowlisted = _allowlisted_bundles()
    referenced = _referenced_bundles()
    assert referenced, "no robot config references a model bundle"

    unfetchable = {
        name: sorted(origins)
        for name, origins in referenced.items()
        if name not in allowlisted and name not in EXTERNALLY_PROVISIONED
    }
    assert not unfetchable, (
        "robot configs reference model bundles that scripts/download_models.py "
        f"cannot fetch: {unfetchable}. Add them to MODEL_BUNDLE_ALLOWLIST, fix the "
        "bundle_path, or record them in EXTERNALLY_PROVISIONED with the route that "
        "supplies them; otherwise the tests needing them skip on every machine."
    )


def test_externally_provisioned_entries_are_still_referenced():
    """Retire the exemption list when a config stops using an entry."""
    referenced = set(_referenced_bundles())
    stale = sorted(EXTERNALLY_PROVISIONED - referenced)
    assert not stale, f"EXTERNALLY_PROVISIONED lists bundles no config references: {stale}"


@pytest.mark.parametrize("name", sorted(_referenced_bundles()))
def test_referenced_bundle_name_is_a_plain_directory_name(name: str):
    """Guard against a path fragment sneaking in where a bundle name belongs."""
    assert name and "/" not in name and not name.startswith("."), f"suspicious bundle name: {name!r}"

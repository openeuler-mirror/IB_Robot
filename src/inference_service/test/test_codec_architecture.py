from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

CODEC_ROOT = Path(__file__).resolve().parents[1] / "inference_service" / "codecs"
SHARED_MODULES = tuple(sorted(CODEC_ROOT.glob("*.py")))
BACKEND_IDENTITIES = {"ascend", "hisilicon", "rknn", "hmm", "torch"}
FORBIDDEN_IMPORT_ROOTS = {
    "acl",
    "rknn",
    "rknnlite",
    "tcim",
    "torch",
    "torch_npu",
}
RUNTIME_RESOURCE_FIELD_PARTS = {
    "address",
    "context",
    "device_buffer",
    "handle",
    "pointer",
    "ptr",
    "sdk",
    "stream",
}


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _import_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _string_literals(tree: ast.Module) -> set[str]:
    values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            values.add(node.value.lower())
    return values


def _declared_fields(tree: ast.Module) -> set[str]:
    fields: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                fields.add(statement.target.id.lower())
    return fields


def test_shared_codec_modules_do_not_import_backend_sdks_or_plugins():
    violations: dict[str, list[str]] = {}
    for path in SHARED_MODULES:
        imports = _import_names(_parse(path))
        forbidden = sorted(
            name
            for name in imports
            if name.split(".", maxsplit=1)[0] in FORBIDDEN_IMPORT_ROOTS
            or any(re.search(rf"(?<![a-z0-9]){identity}(?![a-z0-9])", name.lower()) for identity in BACKEND_IDENTITIES)
        )
        if forbidden:
            violations[path.name] = forbidden

    assert violations == {}


def test_shared_codec_modules_have_no_backend_identity_literals():
    violations: dict[str, list[str]] = {}
    for path in SHARED_MODULES:
        literals = _string_literals(_parse(path))
        forbidden = sorted(
            literal
            for literal in literals
            if any(re.search(rf"(?<![a-z0-9]){identity}(?![a-z0-9])", literal) for identity in BACKEND_IDENTITIES)
        )
        if forbidden:
            violations[path.name] = forbidden

    assert violations == {}


def test_shared_codec_objects_cannot_carry_sdk_handles_or_pointer_values():
    violations: dict[str, list[str]] = {}
    for path in SHARED_MODULES:
        fields = _declared_fields(_parse(path))
        forbidden = sorted(
            field
            for field in fields
            if field != "transport"
            and any(field == part or field.endswith(f"_{part}") for part in RUNTIME_RESOURCE_FIELD_PARTS)
        )
        if forbidden:
            violations[path.name] = forbidden

    assert violations == {}


def test_authoritative_codec_registry_selects_by_policy_metadata_only():
    from inference_service.codecs import POLICY_CODEC_REGISTRY, create_policy_codec

    assert POLICY_CODEC_REGISTRY.policy_types == ("act", "pi05", "smolvla")
    assert tuple(inspect.signature(create_policy_codec).parameters) == ("metadata",)


def test_shared_codec_selection_has_no_allowlists_or_output_identity_branches():
    violations: dict[str, list[int]] = {}
    for path in SHARED_MODULES:
        tree = _parse(path)
        lines: list[int] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            compared_literals = {
                item.value.lower()
                for item in (node.left, *node.comparators)
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            }
            if compared_literals & BACKEND_IDENTITIES:
                lines.append(node.lineno)
        if lines:
            violations[path.name] = lines

    assert violations == {}

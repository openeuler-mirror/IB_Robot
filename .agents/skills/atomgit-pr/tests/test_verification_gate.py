import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import verification_gate
from verification_gate import (
    compute_verification_inputs,
    file_triggers_dual_docker_gate,
    format_verification_metadata,
    is_wip_title,
    normalize_pr_title,
    prepare_update_verification,
    resolve_pr_head_tree,
    resolve_pr_stage,
    upsert_verification_metadata,
    validate_verification_metadata,
)

_ENV = "ubuntu:ros-humble-desktop-full-jammy|openeuler:ibrobot-dev-env|policy:1"


def _gated_files():
    return [{"filename": "scripts/setup.sh", "patch": "@@ -1 +1 @@\n-old\n+new"}]


def _inputs_sha():
    return compute_verification_inputs(_gated_files())


def _full_block(tree, inputs):
    return format_verification_metadata("full", inputs, tree, _ENV)


def _reused_block(old_tree, inputs):
    return format_verification_metadata("reused-environment", inputs, old_tree, _ENV)


def test_verification_upsert_is_idempotent_and_preserves_results():
    tree, inputs = "a" * 40, _inputs_sha()
    description = (
        f"## Summary\n\nChange.\n\n{_full_block(tree, inputs)}\n\n### Ubuntu\n\nPassed.\n\n## Risks\n\nNone.\n"
    )
    updated = upsert_verification_metadata(description, "full", inputs, tree, _ENV)
    assert updated == description
    assert upsert_verification_metadata(updated, "full", inputs, tree, _ENV) == updated
    assert updated.count("## Docker Verification") == 1
    validate_verification_metadata(updated, inputs, tree)


def test_verification_upsert_reuses_heading_without_metadata():
    tree, inputs = "a" * 40, _inputs_sha()
    description = "## Docker Verification\n\n### Ubuntu\n\nPassed.\n"
    updated = upsert_verification_metadata(description, "full", inputs, tree, _ENV)
    assert updated == _full_block(tree, inputs) + "\n\n### Ubuntu\n\nPassed.\n"


def test_verification_rejects_duplicate_headings_with_only_one_field_set():
    tree, inputs = "a" * 40, _inputs_sha()
    description = "## Docker Verification\n\nPlatform results.\n\n" + _full_block(tree, inputs)
    with pytest.raises(ValueError, match="exactly one.*heading"):
        validate_verification_metadata(description, inputs, tree)
    with pytest.raises(ValueError, match="exactly one.*heading"):
        upsert_verification_metadata(description, "full", inputs, tree, _ENV)


def test_gate_detects_global_and_package_dependency_changes():
    assert file_triggers_dual_docker_gate("scripts/setup.sh")
    assert file_triggers_dual_docker_gate("requirements/ubuntu.txt")
    assert file_triggers_dual_docker_gate("src/demo/package.xml", "+  <exec_depend>demo</exec_depend>")
    assert not file_triggers_dual_docker_gate("src/demo/package.xml", "+  <description>Updated</description>")
    assert file_triggers_dual_docker_gate("src/demo/package.xml")
    assert not file_triggers_dual_docker_gate("src/demo/setup.py")


def test_pr_creation_requires_description_file():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "scripts" / "pr_creation.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "--description-file" in result.stdout
    assert "--body" not in result.stdout


def test_wip_title_normalization_is_explicit_and_reversible():
    assert normalize_pr_title("agents: update workflow", "wip") == "[WIP] agents: update workflow"
    assert normalize_pr_title("[wip] [WIP] agents: update workflow", "review") == "agents: update workflow"
    assert is_wip_title("[WIP] agents: update workflow")
    assert is_wip_title(" [wip] agents: update workflow")
    assert not is_wip_title("agents: update [WIP] workflow")

    with pytest.raises(ValueError, match="must not be empty"):
        normalize_pr_title("[WIP]", "review")


def test_gate_requires_explicit_stage_and_wip_only_defers_docker():
    with pytest.raises(ValueError, match="explicit PR stage"):
        resolve_pr_stage("agents: update workflow", None, True)

    wip_title, wip_gate = resolve_pr_stage("agents: update workflow", "wip", True)
    review_title, review_gate = resolve_pr_stage(wip_title, "review", True)

    assert wip_title == "[WIP] agents: update workflow"
    assert not wip_gate
    assert review_title == "agents: update workflow"
    assert review_gate


def test_compute_verification_inputs_is_deterministic():
    sha1 = compute_verification_inputs(_gated_files())
    sha2 = compute_verification_inputs(_gated_files())
    assert sha1 == sha2
    assert len(sha1) == 40


def test_compute_verification_inputs_changes_with_patch():
    files_a = [{"filename": "scripts/setup.sh", "patch": "@@ -1 +1 @@\n-old\n+new"}]
    files_b = [{"filename": "scripts/setup.sh", "patch": "@@ -1 +1 @@\n-old\n+other"}]
    assert compute_verification_inputs(files_a) != compute_verification_inputs(files_b)


def test_compute_verification_inputs_ignores_non_gated_files():
    assert compute_verification_inputs([{"filename": "src/foo.py", "patch": "+x"}]) is None


def test_full_verification_metadata_validates():
    tree = "a" * 40
    inputs = _inputs_sha()
    body = _full_block(tree, inputs)
    metadata = validate_verification_metadata(body, inputs, tree)
    assert metadata["mode"] == "full"
    assert metadata["tested_tree"] == tree


def test_full_verification_rejects_mismatched_tree():
    inputs = _inputs_sha()
    body = _full_block("b" * 40, inputs)
    with pytest.raises(ValueError, match="does not match current tree"):
        validate_verification_metadata(body, inputs, "a" * 40)


def test_reused_environment_validates_with_prior_evidence():
    old_tree = "a" * 40
    new_tree = "b" * 40
    inputs = _inputs_sha()
    body = _reused_block(old_tree, inputs)
    metadata = validate_verification_metadata(body, inputs, new_tree, allow_reuse=True)
    assert metadata["mode"] == "reused-environment"
    assert metadata["tested_tree"] == old_tree


def test_reused_environment_rejects_without_prior_evidence():
    inputs = _inputs_sha()
    body = _reused_block("a" * 40, inputs)
    with pytest.raises(ValueError, match="only valid when prior evidence"):
        validate_verification_metadata(body, inputs, "b" * 40)


def test_prepare_update_carries_forward_reused_environment():
    old_tree = "a" * 40
    new_tree = "b" * 40
    inputs = _inputs_sha()
    previous = _reused_block(old_tree, inputs)
    description = "## Changes\n\nFix a typo.\n"
    updated, metadata = prepare_update_verification(description, previous, inputs, new_tree)
    assert metadata["mode"] == "reused-environment"
    assert metadata["tested_tree"] == old_tree
    assert "reused-environment" in updated


def test_prepare_update_requires_full_when_inputs_change():
    old_tree = "a" * 40
    new_tree = "b" * 40
    inputs_old = _inputs_sha()
    inputs_new = compute_verification_inputs(
        [{"filename": "scripts/setup.sh", "patch": "@@ -1 +1 @@\n-old\n+brand-new"}]
    )
    previous = _full_block(old_tree, inputs_old)
    description = "## Changes\n\nNo Docker block yet.\n"
    with pytest.raises(ValueError, match="full Docker verification is required"):
        prepare_update_verification(description, previous, inputs_new, new_tree)


def test_prepare_update_accepts_draft_reused_block():
    old_tree = "a" * 40
    new_tree = "b" * 40
    inputs = _inputs_sha()
    previous = _reused_block(old_tree, inputs)
    draft = _reused_block(old_tree, inputs)
    updated, metadata = prepare_update_verification(draft, previous, inputs, new_tree)
    assert metadata["mode"] == "reused-environment"
    assert metadata["tested_tree"] == old_tree


def test_prepare_update_accepts_new_full_block():
    tree = "a" * 40
    inputs = _inputs_sha()
    body = _full_block(tree, inputs)
    previous = _full_block(tree, inputs)
    updated, metadata = prepare_update_verification(body, previous, inputs, tree)
    assert metadata["mode"] == "full"


def test_upsert_replaces_legacy_verified_tree():
    tree = "a" * 40
    inputs = _inputs_sha()
    legacy = f"## Verification\n\n**Verified tree:** `{tree}`\n"
    updated = upsert_verification_metadata(legacy, "full", inputs, tree, _ENV)
    assert "**Verified tree:**" not in updated
    assert "**Verified inputs:**" in updated


def test_resolve_pr_head_tree_fetches_source_branch(monkeypatch):
    head = "b" * 40
    tree = "c" * 40
    calls = []

    def fake_run_git(args, cwd):
        calls.append(args)
        if args == ["rev-parse", "FETCH_HEAD"]:
            return head
        if args == ["rev-parse", "FETCH_HEAD^{tree}"]:
            return tree
        return ""

    monkeypatch.setattr(verification_gate, "_run_git", fake_run_git)
    pr = {
        "head": {
            "sha": head,
            "ref": "feature/example",
            "repo": {"html_url": "https://atomgit.com/example/IB_Robot.git"},
        }
    }

    assert resolve_pr_head_tree(pr) == tree
    assert [
        "fetch",
        "--no-tags",
        "--depth=1",
        pr["head"]["repo"]["html_url"],
        "refs/heads/feature/example",
    ] in calls


def test_resolve_pr_head_tree_rejects_racing_head(monkeypatch):
    monkeypatch.setattr(
        verification_gate,
        "_run_git",
        lambda args, cwd: "b" * 40 if args == ["rev-parse", "FETCH_HEAD"] else "",
    )
    pr = {
        "head": {
            "sha": "a" * 40,
            "ref": "feature/example",
            "repo": {"html_url": "https://atomgit.com/example/IB_Robot.git"},
        }
    }

    with pytest.raises(ValueError, match="head changed"):
        resolve_pr_head_tree(pr)


def test_resolve_pr_head_tree_rejects_non_https_repo():
    pr = {
        "head": {
            "sha": "a" * 40,
            "ref": "feature/example",
            "repo": {"html_url": "file:///tmp/IB_Robot"},
        }
    }

    with pytest.raises(ValueError, match="HTTPS repository URL"):
        resolve_pr_head_tree(pr)


def test_extract_accepts_list_prefix_fields():
    """Regression: Agents naturally write Markdown bullet lists; the gate must
    accept '- **label:**' and upsert must strip it before appending canonical
    lines, otherwise the description ends up with duplicate blocks."""
    from verification_gate import extract_verification_metadata

    inputs = _inputs_sha()
    tree = "b" * 40
    description = (
        "## Docker Verification\n\n"
        f"- **Docker verification mode:** `full`\n"
        f"- **Verified inputs:** `{inputs}`\n"
        f"- **Tested source tree:** `{tree}`\n"
        f"- **Docker environment:** `ubuntu|openeuler`"
    )

    metadata = extract_verification_metadata(description)
    assert metadata == {
        "mode": "full",
        "verified_inputs": inputs,
        "tested_tree": tree,
        "environment": "ubuntu|openeuler",
    }

    # Upsert must recognise the list-format lines as the same block (not new
    # fields), so no duplicates remain afterwards.
    updated = upsert_verification_metadata(description, "full", inputs, tree, "ubuntu|openeuler")
    import re

    for label in (
        verification_gate.VERIFICATION_MODE_LABEL,
        verification_gate.VERIFIED_INPUTS_LABEL,
        verification_gate.TESTED_SOURCE_TREE_LABEL,
        verification_gate.VERIFICATION_ENVIRONMENT_LABEL,
    ):
        pattern = r"\*\*" + re.escape(label) + r":\*\*"
        total = len(re.findall(pattern, updated))
        list_form = len(re.findall(r"- " + pattern, updated))
        assert total == 1, f"{label} appears {total} times after upsert"
        assert list_form == 0, f"{label} still uses list format after upsert"


def test_mismatch_errors_report_expected_values():
    """Regression: mismatch errors used to hide the expected fingerprint/tree,
    forcing the Agent to monkey-patch internal helpers to discover them."""
    from verification_gate import validate_verification_metadata

    inputs = _inputs_sha()
    good_tree = "b" * 40
    bad_tree = "c" * 40
    block = format_verification_metadata("full", inputs, bad_tree, _ENV)

    with pytest.raises(ValueError, match=good_tree):
        validate_verification_metadata(block, inputs, good_tree)

    other_inputs = compute_verification_inputs(
        [{"filename": "scripts/setup.sh", "patch": "@@ -1 +1 @@\n-different\n+patch"}]
    )
    wrong_inputs_block = format_verification_metadata("full", other_inputs, good_tree, _ENV)
    with pytest.raises(ValueError, match=other_inputs):
        validate_verification_metadata(wrong_inputs_block, inputs, good_tree)


def test_duplicate_field_error_lists_actual_counts():
    from verification_gate import extract_verification_metadata

    inputs = _inputs_sha()
    duplicated = (
        format_verification_metadata("full", inputs, "b" * 40, _ENV) + "\n\n**Docker verification mode:** `full`\n"
    )

    with pytest.raises(ValueError) as excinfo:
        extract_verification_metadata(duplicated)
    message = str(excinfo.value)
    # Only the offending fields are listed, with their actual counts.
    assert "Docker verification mode=2" in message
    assert "Verified inputs" not in message

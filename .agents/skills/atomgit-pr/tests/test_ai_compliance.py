import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from ai_compliance import (
    DISCLOSURE_END,
    DISCLOSURE_START,
    add_ai_disclosure,
    reject_handwritten_disclosure,
    validate_agent_tool,
    validate_commit_ai_model,
)


def test_validate_agent_tool_accepts_arbitrary_tool_report():
    assert validate_agent_tool("Acme Coding Agent 2026.08.19") == "Acme Coding Agent 2026.08.19"
    assert validate_agent_tool("vendor/coding-agent v1.2.3-beta.1") == "vendor/coding-agent v1.2.3-beta.1"


def test_validate_agent_tool_rejects_unversioned_or_injection_text():
    for value in ("Acme Coding Agent latest", "Acme Agent 1.2.3; rm -rf /"):
        with pytest.raises(ValueError, match="actual tool name and version"):
            validate_agent_tool(value)


def test_add_ai_disclosure_records_required_metadata():
    body = add_ai_disclosure(
        "## Changes\n\nFocused update.",
        agent_tool="OpenCode 1.17.20",
        ai_model="gpt-5.6-sol",
        prompt_summary="Apply the requested policy",
        third_party_materials="无",
    )

    assert body.count(DISCLOSURE_START) == 1
    assert "Agent平台信息（Tool）: OpenCode 1.17.20" in body
    assert "模型信息 (Model): gpt-5.6-sol" in body
    assert "人工审查情况" in body
    assert f"{DISCLOSURE_END}\n\n---\n\n## Changes" in body
    assert body.endswith("## Changes\n\nFocused update.")


def test_add_ai_disclosure_separates_block_from_body_with_single_rule():
    body = add_ai_disclosure(
        "---\n\n## Changes\n\nBody already starts with a rule.",
        agent_tool="OpenCode 1.17.20",
        ai_model="gpt-5.6-sol",
        prompt_summary="Apply the requested policy",
        third_party_materials="无",
    )

    assert body.count("\n---\n") == 1
    assert f"{DISCLOSURE_END}\n\n---\n\n## Changes" in body


def test_add_ai_disclosure_replaces_existing_block():
    first = add_ai_disclosure(
        "Description",
        agent_tool="OpenCode 1.17.20",
        ai_model="model-v1",
        prompt_summary="Initial prompt",
        third_party_materials="无",
    )
    updated = add_ai_disclosure(
        first,
        agent_tool="Claude Code 2.1.223",
        ai_model="model-v2",
        prompt_summary="Updated prompt",
        third_party_materials="无",
    )

    assert updated.count(DISCLOSURE_START) == 1
    assert "OpenCode 1.17.20" not in updated
    assert "模型信息 (Model): model-v2" in updated
    assert "Agent平台信息（Tool）: Claude Code 2.1.223" in updated
    assert f"{DISCLOSURE_END}\n\n---\n\nDescription" in updated


def test_reject_handwritten_disclosure_outside_managed_block():
    managed = add_ai_disclosure(
        "## Changes\n\nBody.",
        agent_tool="OpenCode 1.17.20",
        ai_model="gpt-5.6-sol",
        prompt_summary="Apply the policy",
        third_party_materials="无",
    )
    # The managed block itself must not trigger the rejection.
    reject_handwritten_disclosure(managed)

    for handwritten in (
        "# AI 披露\n\n1. Agent平台信息（Tool）: OpenCode 1.17.20",
        "### 当前PR是否有AI参与\n- [x] 是",
        "### 希望检视人员了解\n内容由 AI 辅助完成",
        managed + "\n\n## AI 披露\n\n1. Agent平台信息（Tool）: OpenCode 1.17.20",
    ):
        with pytest.raises(ValueError, match="hand-written AI disclosure"):
            reject_handwritten_disclosure(handwritten)


def test_validate_commit_ai_model_accepts_matching_trailers():
    commits = [
        {"hash": "0" * 40, "subject": "docs: human authored context", "body": "No AI trailer."},
        {
            "hash": "a" * 40,
            "subject": "docs: update policy",
            "body": "Explain the policy.\n\nCo-Authored-By: gpt-5.6-sol\nSigned-off-by: User <user@example.com>",
        },
    ]

    validate_commit_ai_model(commits, "gpt-5.6-sol")


def test_validate_commit_ai_model_accepts_multiple_models_and_human_coauthor():
    commits = [
        {"sha": "a" * 40, "commit": {"message": "feat: one\n\nCo-Authored-By: gpt-5.6-sol"}},
        {
            "sha": "b" * 40,
            "commit": {
                "message": "feat: two\n\nCo-Authored-By: DeepSeek-V3\n"
                "Co-Authored-By: Human Contributor <human@example.com>"
            },
        },
    ]

    validate_commit_ai_model(commits, "gpt-5.6-sol, DeepSeek-V3")


def test_add_ai_disclosure_deduplicates_multiple_models():
    body = add_ai_disclosure(
        "Description",
        agent_tool="OpenCode 1.17.20",
        ai_model="gpt-5.6-sol, DeepSeek-V3, gpt-5.6-sol",
        prompt_summary="Apply the policy",
        third_party_materials="无",
    )

    assert "模型信息 (Model): gpt-5.6-sol, DeepSeek-V3" in body


def test_validate_commit_ai_model_rejects_missing_disclosed_model():
    commits = [
        {"sha": "a" * 40, "commit": {"message": "docs: missing trailer"}},
        {"sha": "b" * 40, "commit": {"message": "docs: mismatch\n\nCo-Authored-By: other-model"}},
    ]

    with pytest.raises(ValueError, match="union of every commit's Co-Authored-By model") as excinfo:
        validate_commit_ai_model(commits, "gpt-5.6-sol")
    assert "other-model" in str(excinfo.value)


def test_ai_model_rejects_provider_prefix():
    with pytest.raises(ValueError, match="must not include a provider prefix"):
        add_ai_disclosure(
            "Description",
            agent_tool="OpenCode 1.17.20",
            ai_model="xunxing/gpt-5.6-sol",
            prompt_summary="Apply the policy",
            third_party_materials="无",
        )


def test_add_ai_disclosure_rejects_unversioned_agent_tool():
    with pytest.raises(ValueError, match="actual tool name and version"):
        add_ai_disclosure(
            "Description",
            agent_tool="OpenCode latest",
            ai_model="gpt-5.6-sol",
            prompt_summary="Apply the policy",
            third_party_materials="无",
        )

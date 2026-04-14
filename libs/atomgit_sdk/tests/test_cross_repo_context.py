#!/usr/bin/env python3
"""Tests for cross-repo AtomGit context resolution."""

import json

from atomgit_sdk import AtomGitConfig, parse_atomgit_url, resolve_atomgit_context


def write_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "atomgit": {
                    "token": "test-token",
                    "owner": "example-org",
                    "repo": "demo-repo",
                    "baseUrl": "https://api.atomgit.com",
                }
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_parse_atomgit_pr_url():
    parsed = parse_atomgit_url("https://atomgit.com/foo/bar/pull/12")
    assert parsed == {"owner": "foo", "repo": "bar", "pr_number": 12}


def test_parse_atomgit_issue_url():
    parsed = parse_atomgit_url("https://atomgit.com/foo/bar/issues/34")
    assert parsed == {"owner": "foo", "repo": "bar", "issue_number": 34}


def test_parse_gitcode_merge_request_url():
    parsed = parse_atomgit_url("https://gitcode.com/foo/bar/merge_requests/56")
    assert parsed == {"owner": "foo", "repo": "bar", "pr_number": 56}


def test_from_json_applies_url_overrides(tmp_path):
    config_path = write_config(tmp_path)
    config = AtomGitConfig.from_json(
        str(config_path), url="https://atomgit.com/other/repo/pull/99"
    )
    assert config.owner == "other"
    assert config.repo == "repo"
    assert config.token == "test-token"


def test_resolve_atomgit_context_prefers_explicit_owner_repo(tmp_path):
    config_path = write_config(tmp_path)
    config, parsed = resolve_atomgit_context(
        str(config_path),
        owner="explicit-owner",
        repo="explicit-repo",
        url="https://atomgit.com/url-owner/url-repo/issues/42",
    )
    assert parsed == {"owner": "url-owner", "repo": "url-repo", "issue_number": 42}
    assert config.owner == "explicit-owner"
    assert config.repo == "explicit-repo"

#!/usr/bin/env python3
"""Tests for AtomGit API catalog utilities."""

import pytest
from atomgit_sdk import DEFAULT_API_CATALOG
from atomgit_sdk.exceptions import ConfigurationError


def test_default_catalog_contains_pr_discussion_endpoints():
    reply_endpoint = DEFAULT_API_CATALOG.get(
        "post-api-v-5-repos-owner-repo-pulls-number-discussions-discussions-id-comments"
    )
    assert reply_endpoint.method == "POST"
    assert reply_endpoint.path == "/api/v5/repos/:owner/:repo/pulls/:number/discussions/:discussion_id/comments"

    rendered = reply_endpoint.render(
        {
            "owner": "openEuler",
            "repo": "IB_Robot",
            "number": 32,
            "discussion_id": "abc",
        }
    )
    assert rendered == "/api/v5/repos/openEuler/IB_Robot/pulls/32/discussions/abc/comments"


def test_catalog_search_finds_pull_comment_routes():
    endpoints = DEFAULT_API_CATALOG.find(path_contains="/pulls/comments")

    slugs = {endpoint.slug for endpoint in endpoints}
    assert "get-api-v-5-repos-owner-repo-pulls-comments-id" in slugs
    assert "patch-api-v-5-repos-owner-repo-pulls-comments-id" in slugs
    assert "delete-api-v-5-repos-owner-repo-pulls-comments-id" in slugs


def test_endpoint_render_requires_all_path_params():
    endpoint = DEFAULT_API_CATALOG.get("get-api-v-5-repos-owner-repo-pulls-number")

    with pytest.raises(ConfigurationError, match="Missing path parameters"):
        endpoint.render({"owner": "openEuler", "repo": "IB_Robot"})

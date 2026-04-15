#!/usr/bin/env python3
"""Tests for AtomGit issue label validation."""

from unittest.mock import Mock

import pytest

from atomgit_sdk import AtomGitClient, AtomGitConfig
from atomgit_sdk.exceptions import AtomGitAPIError


def make_client():
    config = AtomGitConfig(
        token="test-token",
        owner="example-org",
        repo="example-repo",
        base_url="https://api.atomgit.com",
    )
    return AtomGitClient(config)


def test_create_issue_rejects_unknown_labels(monkeypatch):
    client = make_client()
    request_mock = Mock(return_value=[{"name": "bug"}])
    monkeypatch.setattr(client, "request", request_mock)

    with pytest.raises(AtomGitAPIError, match="Unknown labels"):
        client.create_issue("test title", "test body", labels=["enhancement"])

    assert request_mock.call_count == 1


def test_create_issue_accepts_existing_labels(monkeypatch):
    client = make_client()
    request_mock = Mock(
        side_effect=[
            [{"name": "enhancement"}],
            {"number": 9},
        ]
    )
    monkeypatch.setattr(client, "request", request_mock)

    result = client.create_issue("test title", "test body", labels=["enhancement"])

    assert result == {"number": 9}
    assert request_mock.call_count == 2

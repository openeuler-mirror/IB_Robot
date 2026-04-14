#!/usr/bin/env python3
"""Tests for AtomGit comment retrieval services."""

from unittest.mock import Mock

from atomgit_sdk.services import IssueService, PRService


def test_issue_service_get_issue_comments():
    mock_client = Mock()
    mock_client.get_issue_comments.return_value = [{"id": 1, "body": "hello"}]

    service = IssueService(mock_client)

    result = service.get_issue_comments(31)

    mock_client.get_issue_comments.assert_called_once_with(31)
    assert result == [{"id": 1, "body": "hello"}]


def test_pr_service_get_pr_comments():
    mock_client = Mock()
    mock_client.get_pr_comments.return_value = [{"id": 2, "body": "world"}]

    service = PRService(mock_client)

    result = service.get_pr_comments(70)

    mock_client.get_pr_comments.assert_called_once_with(70)
    assert result == [{"id": 2, "body": "world"}]


def test_pr_service_full_context_includes_comments():
    mock_client = Mock()
    mock_client.get_pull_request.return_value = {"number": 70}
    mock_client.get_pr_files.return_value = [{"filename": "a.py"}]
    mock_client.get_pr_commits.return_value = [{"sha": "abc"}]
    mock_client.get_pr_diff.return_value = {"a.py": {"patch": "@@ -1 +1 @@"}}
    mock_client.get_pr_comments.return_value = [{"id": 3, "body": "looks good"}]

    service = PRService(mock_client)

    context = service.get_full_pr_context(70)

    assert context["comments"] == [{"id": 3, "body": "looks good"}]

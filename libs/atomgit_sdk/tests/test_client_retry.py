#!/usr/bin/env python3
"""Tests for AtomGit client retry behavior."""

from unittest.mock import Mock

import pytest
import requests
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


def test_get_retries_on_ssl_error(monkeypatch):
    client = make_client()
    success_response = Mock(status_code=200)
    success_response.json.return_value = {"number": 63}

    request_mock = Mock(
        side_effect=[
            requests.exceptions.SSLError("EOF occurred in violation of protocol"),
            success_response,
        ]
    )
    monkeypatch.setattr(client.session, "request", request_mock)
    monkeypatch.setattr("atomgit_sdk.client.time.sleep", lambda _: None)

    result = client.request("GET", "/api/v5/repos/example-org/example-repo/pulls/63")

    assert result == {"number": 63}
    assert request_mock.call_count == 2


def test_post_does_not_retry_on_ssl_error(monkeypatch):
    client = make_client()
    request_mock = Mock(side_effect=requests.exceptions.SSLError("EOF occurred in violation of protocol"))
    monkeypatch.setattr(client.session, "request", request_mock)

    with pytest.raises(AtomGitAPIError, match="Request failed"):
        client.request(
            "POST",
            "/api/v5/repos/example-org/example-repo/pulls",
            body={"title": "test"},
        )

    assert request_mock.call_count == 1


def test_request_passes_query_params(monkeypatch):
    client = make_client()
    response = Mock(status_code=200)
    response.json.return_value = []
    request_mock = Mock(return_value=response)
    monkeypatch.setattr(client.session, "request", request_mock)

    result = client.request("GET", "/api/v5/user/repos", params={"page": 2})

    assert result == []
    request_mock.assert_called_once()
    assert request_mock.call_args.kwargs["params"] == {"page": 2}


def test_submit_inline_comment_uses_new_line_without_position(monkeypatch):
    client = make_client()
    response = Mock(status_code=200)
    response.json.return_value = {"id": 99}
    request_mock = Mock(return_value=response)
    monkeypatch.setattr(client.session, "request", request_mock)

    result = client.submit_inline_comment(
        12,
        {"path": "src/main.py", "new_line": 42, "body": "check this"},
    )

    assert result == {"id": 99}
    payload = request_mock.call_args.kwargs["json"]
    assert payload["new_line"] == 42
    assert "position" not in payload


def test_pr_discussion_reply_endpoint(monkeypatch):
    client = make_client()
    response = Mock(status_code=200)
    response.json.return_value = {"id": 101}
    request_mock = Mock(return_value=response)
    monkeypatch.setattr(client.session, "request", request_mock)

    result = client.reply_to_pr_discussion(12, "disc-1", "reply")

    assert result == {"id": 101}
    assert (
        request_mock.call_args.kwargs["url"]
        == "https://api.atomgit.com/api/v5/repos/example-org/example-repo/pulls/12/discussions/disc-1/comments"
    )
    assert request_mock.call_args.kwargs["json"] == {"body": "reply"}


def test_pr_discussion_resolve_endpoint(monkeypatch):
    client = make_client()
    response = Mock(status_code=200)
    response.json.return_value = {"resolved": True}
    request_mock = Mock(return_value=response)
    monkeypatch.setattr(client.session, "request", request_mock)

    result = client.set_pr_discussion_resolved(12, "disc-1", True)

    assert result == {"resolved": True}
    assert (
        request_mock.call_args.kwargs["url"]
        == "https://api.atomgit.com/api/v5/repos/example-org/example-repo/pulls/12/comments/disc-1"
    )
    assert request_mock.call_args.kwargs["json"] == {"resolved": True}

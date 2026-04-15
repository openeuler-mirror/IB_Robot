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
            requests.exceptions.SSLError(
                "EOF occurred in violation of protocol"
            ),
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
    request_mock = Mock(
        side_effect=requests.exceptions.SSLError(
            "EOF occurred in violation of protocol"
        )
    )
    monkeypatch.setattr(client.session, "request", request_mock)

    with pytest.raises(AtomGitAPIError, match="Request failed"):
        client.request(
            "POST",
            "/api/v5/repos/example-org/example-repo/pulls",
            body={"title": "test"},
        )

    assert request_mock.call_count == 1

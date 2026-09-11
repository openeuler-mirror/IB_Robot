from __future__ import annotations

import json

import pytest
from ibrobot_agent.node import _planner_from_config


def test_kimicode_planner_uses_environment_key(monkeypatch):
    monkeypatch.setenv("KIMICODE_API_KEY", "test-kimi-key")

    planner = _planner_from_config(
        json.dumps(
            {
                "mode": "vlm",
                "provider": "kimicode",
                "base_url": "https://api.kimi.com/coding/v1",
                "api_key_env": "KIMICODE_API_KEY",
                "model": "kimi-for-coding",
            }
        )
    )

    assert planner.identity.route == "vlm"
    assert planner.identity.returned_model == "kimi-for-coding"
    assert planner._model_client._build_headers() == {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-kimi-key",
        "User-Agent": "KimiCLI/1.3",
    }


def test_kimi_coding_configuration_requires_temperature_one():
    from robot_config.loader import validate_agent_entry_config

    errors = validate_agent_entry_config(
        {
            "entry_mode": "agent",
            "agent": {
                "enabled": True,
                "incubation": True,
                "test_allowlist": ["nod_yes"],
                "ledger_path": "/tmp/request.sqlite3",
                "conversation_path": "/tmp/conversation.sqlite3",
                "deployment_lock_path": "/tmp/agent.lock",
                "planner": {
                    "mode": "vlm",
                    "provider": "kimicode",
                    "base_url": "https://api.kimi.com/coding/v1",
                    "api_key_env": "KIMICODE_API_KEY",
                    "model": "kimi-for-coding",
                    "temperature": 0.0,
                },
            },
        }
    )

    assert "embodied.agent.planner.temperature must be 1 for kimi-for-coding" in errors


def test_vlm_planner_rejects_literal_api_key():
    with pytest.raises(ValueError, match="must not carry a literal api_key"):
        _planner_from_config(
            json.dumps(
                {
                    "mode": "vlm",
                    "provider": "openai_compatible",
                    "base_url": "https://example.invalid/v1",
                    "api_key": "sk-literal-secret",
                    "model": "test-model",
                }
            )
        )


def test_agent_entry_config_rejects_literal_planner_api_key():
    from robot_config.loader import validate_agent_entry_config

    errors = validate_agent_entry_config(
        {
            "entry_mode": "agent",
            "agent": {
                "enabled": True,
                "incubation": True,
                "test_allowlist": ["nod_yes"],
                "ledger_path": "/tmp/request.sqlite3",
                "conversation_path": "/tmp/conversation.sqlite3",
                "deployment_lock_path": "/tmp/agent.lock",
                "planner": {
                    "mode": "vlm",
                    "provider": "kimicode",
                    "base_url": "https://api.kimi.com/coding/v1",
                    "api_key_env": "KIMICODE_API_KEY",
                    "api_key": "sk-literal-secret",
                    "model": "kimi-for-coding",
                },
            },
        }
    )

    assert "embodied.agent.planner must not contain a literal api_key; use api_key_env instead" in errors

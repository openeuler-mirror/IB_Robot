"""Execution admission modes for Hermes Agent plans."""

from __future__ import annotations

INTERACTIVE_CONFIRMATION = "interactive_confirmation"
IMMEDIATE_AFTER_PRESENTATION = "immediate_after_presentation"
AGENT_EXECUTION_MODES = frozenset({INTERACTIVE_CONFIRMATION, IMMEDIATE_AFTER_PRESENTATION})


def validate_agent_execution_mode(value: str) -> str:
    mode = str(value).strip() or INTERACTIVE_CONFIRMATION
    if mode not in AGENT_EXECUTION_MODES:
        raise ValueError(f"unsupported agent execution mode: {mode}")
    return mode

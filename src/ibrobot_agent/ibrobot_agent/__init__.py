"""Lightweight natural-language Agent package for IB-Robot.

The initial package is intentionally a non-operational skeleton. Runtime motion
must remain behind the existing robot-skill and Capability Gateway contracts.
"""

__version__ = "0.1.0"
from .service import AcceptedResponse, AgentService, InMemoryConversationStore

__all__ = ["AcceptedResponse", "AgentService", "InMemoryConversationStore"]

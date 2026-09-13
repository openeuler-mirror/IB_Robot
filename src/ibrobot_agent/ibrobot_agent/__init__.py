"""Natural-language Agent incubation runtime for IB-Robot.

All robot execution is delegated to the existing Capability Gateway chain;
this package owns no skill catalog, motion authorization, or physical
execution.
"""

__version__ = "0.1.0"
from .service import AcceptedResponse, AgentService, InMemoryConversationStore

__all__ = ["AcceptedResponse", "AgentService", "InMemoryConversationStore"]

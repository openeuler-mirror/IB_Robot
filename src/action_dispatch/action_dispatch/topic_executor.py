"""Backward-compatible shim.

The real implementation now lives in ``action_dispatch.executors.topic``. This
module remains so legacy imports ``from action_dispatch.topic_executor import
TopicExecutor`` keep working and return the same class object.
"""

from action_dispatch.executors.topic import TopicExecutor

__all__ = ["TopicExecutor"]

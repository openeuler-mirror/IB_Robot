"""Last-mile ActionExecutor subpackage.

executor registry contract boundary: this subpackage only owns the synchronous executor contract and
the production registry that selects the last-mile action output.

completion-aware executor contract adds the frozen completion envelope (``ExecutionContext``,
``ExecutionReceipt``, ``ExecutionCompletion``, ``CompletionStatus``) and evolves
``ActionExecutor`` to a v2 contract with ``submit`` + ``drain_completions``.
It must not define scheduler state machines, benchmark step service clients or
episode reset barriers; those belong to later Work Packages or to the
``schedulers`` sibling subpackage.
"""

from .base import ActionExecutor
from .completion import (
    CompletionStatus,
    ExecutionCompletion,
    ExecutionContext,
    ExecutionReceipt,
)

__all__ = [
    "ActionExecutor",
    "CompletionStatus",
    "ExecutionContext",
    "ExecutionCompletion",
    "ExecutionReceipt",
]

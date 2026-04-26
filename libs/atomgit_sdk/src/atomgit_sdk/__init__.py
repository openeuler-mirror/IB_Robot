"""
AtomGit SDK

Unified Python SDK for AtomGit/GitCode API operations, including PR
management, issue workflows, code review, and repair services.
"""

__version__ = "0.1.0"

from atomgit_sdk.api_catalog import DEFAULT_API_CATALOG, APICatalog, APIEndpoint
from atomgit_sdk.client import AtomGitClient
from atomgit_sdk.config import AtomGitConfig, resolve_atomgit_context
from atomgit_sdk.exceptions import (
    AtomGitAPIError,
    AtomGitSDKError,
    ConfigurationError,
    DiffParseError,
    URLError,
)
from atomgit_sdk.models import ArchitectureIssue, BaseIssue, CodeIssue, FixResult
from atomgit_sdk.services.issue_service import IssueService
from atomgit_sdk.services.pr_service import PRService
from atomgit_sdk.utils import parse_atomgit_url

__all__ = [
    "AtomGitClient",
    "AtomGitConfig",
    "APIEndpoint",
    "APICatalog",
    "DEFAULT_API_CATALOG",
    "resolve_atomgit_context",
    "parse_atomgit_url",
    "BaseIssue",
    "CodeIssue",
    "ArchitectureIssue",
    "FixResult",
    "PRService",
    "IssueService",
    "AtomGitSDKError",
    "AtomGitAPIError",
    "ConfigurationError",
    "DiffParseError",
    "URLError",
]

"""
Issue Service - High-level issue operations
"""

from typing import Dict, List, Optional
from atomgit_sdk.client import AtomGitClient


class IssueService:
    """High-level issue operations service"""

    def __init__(self, client: AtomGitClient):
        self.client = client

    def get_issues(self, state: str = "open") -> List[dict]:
        """Get list of issues"""
        return self.client.get_issues(state)

    def get_issue(self, issue_number: int) -> dict:
        """Get issue details"""
        return self.client.get_issue(issue_number)

    def get_issue_comments(self, issue_number: int) -> List[dict]:
        """Get issue comments"""
        return self.client.get_issue_comments(issue_number)

    def create_issue(
        self,
        title: str,
        body: str = "",
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
    ) -> dict:
        """Create new issue"""
        return self.client.create_issue(title, body, labels, assignees)

    def update_issue(
        self,
        issue_number: int,
        title: Optional[str] = None,
        body: Optional[str] = None,
        state: Optional[str] = None,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
    ) -> dict:
        """Update existing issue"""
        return self.client.update_issue(issue_number, title, body, state, labels, assignees)

    def get_issue_url(self, issue_number: int) -> str:
        """Get issue URL"""
        return self.client.get_issue_url(issue_number)

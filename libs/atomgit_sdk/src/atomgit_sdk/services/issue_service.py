"""
Issue Service - High-level issue operations
"""

from atomgit_sdk.client import AtomGitClient


class IssueService:
    """High-level issue operations service"""

    def __init__(self, client: AtomGitClient):
        self.client = client

    def get_issues(self, state: str = "open") -> list[dict]:
        """Get list of issues"""
        return self.client.get_issues(state)

    def get_issue(self, issue_number: int) -> dict:
        """Get issue details"""
        return self.client.get_issue(issue_number)

    def get_issue_comments(self, issue_number: int) -> list[dict]:
        """Get issue comments"""
        return self.client.get_issue_comments(issue_number)

    def submit_issue_comment(self, issue_number: int, body: str) -> dict:
        """Submit an issue comment"""
        return self.client.submit_issue_comment(issue_number, body)

    def get_issue_comment(self, comment_id: int) -> dict:
        """Get one issue comment"""
        return self.client.get_issue_comment(comment_id)

    def edit_issue_comment(self, comment_id: int, body: str) -> dict:
        """Edit one issue comment"""
        return self.client.edit_issue_comment(comment_id, body)

    def delete_issue_comment(self, comment_id: int) -> dict:
        """Delete one issue comment"""
        return self.client.delete_issue_comment(comment_id)

    def create_issue(
        self,
        title: str,
        body: str = "",
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
    ) -> dict:
        """Create new issue"""
        return self.client.create_issue(title, body, labels, assignees)

    def update_issue(
        self,
        issue_number: int,
        title: str | None = None,
        body: str | None = None,
        state: str | None = None,
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
    ) -> dict:
        """Update existing issue"""
        return self.client.update_issue(issue_number, title, body, state, labels, assignees)

    def get_issue_url(self, issue_number: int) -> str:
        """Get issue URL"""
        return self.client.get_issue_url(issue_number)

"""
AtomGit API Client
"""

import base64
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote as url_quote
import requests
from atomgit_sdk.config import AtomGitConfig
from atomgit_sdk.exceptions import AtomGitAPIError


class AtomGitClient:
    """AtomGit API Client"""

    def __init__(
        self, config: AtomGitConfig, user_agent: str = "AtomGit-SDK/1.0"
    ):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {config.token}",
                "Content-Type": "application/json",
                "User-Agent": user_agent,
            }
        )

    @staticmethod
    def _is_safe_to_retry(method: str) -> bool:
        """Return whether a request method is safe to retry on transport errors."""
        return method.upper() in {"GET", "HEAD", "OPTIONS"}

    def request(
        self,
        method: str,
        endpoint: str,
        body: Optional[Dict] = None,
        retry_count: int = 3,
    ) -> Any:
        """
        Send HTTP request to AtomGit API.

        Args:
            method: HTTP method (GET, POST, PATCH, DELETE)
            endpoint: API endpoint (e.g., /api/v5/repos/...)
            body: Request body for POST/PATCH
            retry_count: Number of retries on transient failures

        Returns:
            API response data

        Raises:
            AtomGitAPIError: If request fails after retries
        """
        url = f"{self.config.base_url}{endpoint}"

        for attempt in range(retry_count):
            try:
                response = self.session.request(
                    method=method, url=url, json=body, timeout=30
                )

                if response.status_code in (200, 201):
                    try:
                        return response.json()
                    except:
                        return {"data": response.text}

                if response.status_code >= 500 and attempt < retry_count - 1:
                    continue

                raise AtomGitAPIError(
                    f"API request failed",
                    status_code=response.status_code,
                    response_body=response.text,
                )

            except requests.exceptions.Timeout:
                if attempt < retry_count - 1:
                    continue
                raise AtomGitAPIError(f"Request timeout after {retry_count} attempts")
            except (
                requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
            ) as e:
                if self._is_safe_to_retry(method) and attempt < retry_count - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise AtomGitAPIError(f"Request failed: {str(e)}")
            except requests.exceptions.RequestException as e:
                raise AtomGitAPIError(f"Request failed: {str(e)}")

    def get_pull_requests(self, state: str = "open") -> List[dict]:
        """Get list of pull requests"""
        return self.request(
            "GET",
            f"/api/v5/repos/{self.config.owner}/{self.config.repo}/pulls?state={state}&per_page=100",
        )

    def get_pull_request(self, pr_number: int) -> dict:
        """Get pull request details"""
        return self.request(
            "GET",
            f"/api/v5/repos/{self.config.owner}/{self.config.repo}/pulls/{pr_number}",
        )

    def get_pr_files(self, pr_number: int) -> List[dict]:
        """Get pull request files"""
        return self.request(
            "GET",
            f"/api/v5/repos/{self.config.owner}/{self.config.repo}/pulls/{pr_number}/files",
        )

    def get_pr_commits(self, pr_number: int) -> List[dict]:
        """Get pull request commits"""
        return self.request(
            "GET",
            f"/api/v5/repos/{self.config.owner}/{self.config.repo}/pulls/{pr_number}/commits",
        )

    def get_pr_comments(self, pr_number: int) -> List[dict]:
        """Get pull request comments"""
        return self.request(
            "GET",
            f"/api/v5/repos/{self.config.owner}/{self.config.repo}/pulls/{pr_number}/comments",
        )

    def get_pr_diff(self, pr_number: int) -> Dict[str, dict]:
        """Get pull request diff"""
        files = self.get_pr_files(pr_number)
        diffs = {}

        for file in files:
            if file.get("patch"):
                patch_content = (
                    file["patch"].get("diff", file["patch"])
                    if isinstance(file["patch"], dict)
                    else file["patch"]
                )
                diffs[file["filename"]] = {
                    "patch": patch_content,
                    "additions": file.get("additions", 0),
                    "deletions": file.get("deletions", 0),
                    "status": file.get("status", "modified"),
                }

        return diffs

    def get_file_content(self, file_path: str, ref: str = "HEAD") -> str:
        """Get file content"""
        encoded_path = url_quote(file_path, safe="")
        data = self.request(
            "GET",
            f"/api/v5/repos/{self.config.owner}/{self.config.repo}/contents/{encoded_path}?ref={ref}",
        )

        if data.get("content"):
            return base64.b64decode(data["content"]).decode("utf-8")
        return ""

    def submit_inline_comment(self, pr_number: int, comment: dict) -> dict:
        """Submit inline comment to PR"""
        if not comment.get("path"):
            raise AtomGitAPIError("Cannot submit inline comment without path")

        payload = {"body": comment["body"], "path": comment["path"]}

        if comment.get("position") is not None:
            payload["position"] = comment["position"]
            if comment.get("commitId"):
                payload["commit_id"] = comment["commitId"]
        elif comment.get("line"):
            payload["position"] = comment["line"]
            if comment.get("commitId"):
                payload["commit_id"] = comment["commitId"]
        else:
            raise AtomGitAPIError(
                f"Cannot submit inline comment for {comment['path']}: no position or line provided"
            )

        return self.request(
            "POST",
            f"/api/v5/repos/{self.config.owner}/{self.config.repo}/pulls/{pr_number}/comments",
            body=payload,
        )

    def submit_pr_comment(self, pr_number: int, body: str) -> dict:
        """Submit PR-level comment"""
        return self.request(
            "POST",
            f"/api/v5/repos/{self.config.owner}/{self.config.repo}/pulls/{pr_number}/comments",
            body={"body": body},
        )

    def submit_batch_comments(self, pr_number: int, comments: List[dict]) -> List[dict]:
        """Submit batch comments"""
        results = []
        comment_base_url = f"https://atomgit.com/{self.config.owner}/{self.config.repo}/pulls/{pr_number}"

        for comment in comments:
            try:
                result = self.submit_inline_comment(pr_number, comment)
                comment_url = (
                    f"{comment_base_url}#comment-{result.get('id', '')}"
                    if result.get("id")
                    else comment_base_url
                )
                results.append(
                    {
                        "success": True,
                        "comment": comment,
                        "result": result,
                        "comment_url": comment_url,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "success": False,
                        "comment": comment,
                        "error": str(e),
                        "comment_url": None,
                    }
                )

        return results

    def create_pull_request(
        self,
        title: str,
        body: str,
        head: str,
        base: str = "master",
        draft: bool = False,
    ) -> dict:
        """Create pull request"""
        if not title or not head or not base:
            raise AtomGitAPIError(
                "Creating PR requires title, head, and base parameters"
            )

        final_head = head if ":" in head else f"{self.config.owner}:{head}"

        return self.request(
            "POST",
            f"/api/v5/repos/{self.config.owner}/{self.config.repo}/pulls",
            body={
                "title": title,
                "body": body or "",
                "head": final_head,
                "base": base,
                "draft": draft,
            },
        )

    def update_pull_request(
        self,
        pr_number: int,
        title: Optional[str] = None,
        body: Optional[str] = None,
        state: Optional[str] = None,
    ) -> dict:
        """Update pull request"""
        payload = {}
        if title is not None:
            payload["title"] = title
        if body is not None:
            payload["body"] = body
        if state is not None:
            payload["state"] = state

        return self.request(
            "PATCH",
            f"/api/v5/repos/{self.config.owner}/{self.config.repo}/pulls/{pr_number}",
            body=payload,
        )

    def get_pr_url(self, pr_number: int) -> str:
        """Get PR URL"""
        return f"https://atomgit.com/{self.config.owner}/{self.config.repo}/pull/{pr_number}"

    def get_issues(self, state: str = "open") -> List[dict]:
        """Get list of issues"""
        return self.request(
            "GET",
            f"/api/v5/repos/{self.config.owner}/{self.config.repo}/issues?state={state}&per_page=100",
        )

    def get_issue(self, issue_number: int) -> dict:
        """Get issue details"""
        return self.request(
            "GET",
            f"/api/v5/repos/{self.config.owner}/{self.config.repo}/issues/{issue_number}",
        )

    def get_issue_comments(self, issue_number: int) -> List[dict]:
        """Get issue comments"""
        return self.request(
            "GET",
            f"/api/v5/repos/{self.config.owner}/{self.config.repo}/issues/{issue_number}/comments",
        )

    def get_labels(self) -> List[dict]:
        """Get repository labels"""
        return self.request(
            "GET",
            f"/api/v5/repos/{self.config.owner}/{self.config.repo}/labels",
        )

    def _validate_labels_exist(self, labels: List[str]) -> None:
        """Ensure all requested labels already exist in the target repository."""
        if isinstance(labels, str):
            requested_labels = [label.strip() for label in labels.split(",") if label.strip()]
        else:
            requested_labels = labels

        existing_labels = {
            label.get("name")
            for label in self.get_labels()
            if isinstance(label, dict) and label.get("name")
        }
        missing_labels = [label for label in requested_labels if label not in existing_labels]
        if missing_labels:
            raise AtomGitAPIError(
                "Unknown labels for target repository: "
                + ", ".join(sorted(missing_labels))
            )

    def create_issue(
        self,
        title: str,
        body: str = "",
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
    ) -> dict:
        """Create issue"""
        payload = {"title": title, "body": body}
        if labels:
            self._validate_labels_exist(labels)
            payload["labels"] = ",".join(labels) if isinstance(labels, list) else labels
        if assignees:
            payload["assignees"] = (
                ",".join(assignees) if isinstance(assignees, list) else assignees
            )

        return self.request(
            "POST",
            f"/api/v5/repos/{self.config.owner}/{self.config.repo}/issues",
            body=payload,
        )

    def update_issue(
        self,
        issue_number: int,
        title: Optional[str] = None,
        body: Optional[str] = None,
        state: Optional[str] = None,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
    ) -> dict:
        """Update issue"""
        payload = {}
        if title is not None:
            payload["title"] = title
        if body is not None:
            payload["body"] = body
        if state is not None:
            payload["state"] = state
        if labels is not None:
            self._validate_labels_exist(labels)
            payload["labels"] = ",".join(labels) if isinstance(labels, list) else labels
        if assignees is not None:
            payload["assignees"] = (
                ",".join(assignees) if isinstance(assignees, list) else assignees
            )

        return self.request(
            "PATCH",
            f"/api/v5/repos/{self.config.owner}/{self.config.repo}/issues/{issue_number}",
            body=payload,
        )

    def get_issue_url(self, issue_number: int) -> str:
        """Get issue URL"""
        return f"https://atomgit.com/{self.config.owner}/{self.config.repo}/issues/{issue_number}"

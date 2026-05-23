"""
AtomGit API Client
"""

import base64
import json
import time
from typing import Any
from urllib.parse import quote as url_quote

import requests

from atomgit_sdk.api_catalog import DEFAULT_API_CATALOG, APIEndpoint
from atomgit_sdk.config import AtomGitConfig
from atomgit_sdk.exceptions import AtomGitAPIError


class AtomGitClient:
    """AtomGit API Client"""

    def __init__(self, config: AtomGitConfig, user_agent: str = "AtomGit-SDK/1.0"):
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
        body: dict | None = None,
        params: dict | None = None,
        retry_count: int = 3,
    ) -> Any:
        """
        Send HTTP request to AtomGit API.

        Args:
            method: HTTP method (GET, POST, PATCH, DELETE)
            endpoint: API endpoint (e.g., /api/v5/repos/...)
            body: Request body for POST/PATCH/PUT
            params: Query parameters
            retry_count: Number of retries on transient failures

        Returns:
            API response data

        Raises:
            AtomGitAPIError: If request fails after retries
        """
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        for attempt in range(retry_count):
            try:
                response = self.session.request(method=method, url=url, json=body, params=params, timeout=30)

                if response.status_code in (200, 201, 202):
                    try:
                        return response.json()
                    except ValueError:
                        return {"data": response.text}
                if response.status_code == 204:
                    return {}

                if response.status_code >= 500 and attempt < retry_count - 1:
                    continue

                raise AtomGitAPIError(
                    "API request failed",
                    status_code=response.status_code,
                    response_body=response.text,
                )

            except requests.exceptions.Timeout as e:
                if attempt < retry_count - 1:
                    continue
                raise AtomGitAPIError(f"Request timeout after {retry_count} attempts") from e
            except (
                requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
            ) as e:
                if self._is_safe_to_retry(method) and attempt < retry_count - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise AtomGitAPIError(f"Request failed: {str(e)}") from e
            except requests.exceptions.RequestException as e:
                raise AtomGitAPIError(f"Request failed: {str(e)}") from e

    def call_api(
        self,
        slug: str,
        *,
        path_params: dict[str, object] | None = None,
        params: dict | None = None,
        body: dict | None = None,
    ) -> Any:
        """Call a documented AtomGit API endpoint by catalog slug."""
        endpoint = DEFAULT_API_CATALOG.get(slug)
        return self.request(
            endpoint.method,
            endpoint.render(path_params),
            params=params,
            body=body,
        )

    def find_api_endpoints(
        self,
        *,
        method: str | None = None,
        path_contains: str | None = None,
        title_contains: str | None = None,
    ) -> list[APIEndpoint]:
        """Search the built-in documented endpoint catalog."""
        return DEFAULT_API_CATALOG.find(
            method=method,
            path_contains=path_contains,
            title_contains=title_contains,
        )

    def _repo_path(self) -> str:
        owner = url_quote(self.config.owner, safe="")
        repo = url_quote(self.config.repo, safe="")
        return f"/api/v5/repos/{owner}/{repo}"

    def get_pull_requests(self, state: str = "open") -> list[dict]:
        """Get list of pull requests"""
        return self.request(
            "GET",
            f"{self._repo_path()}/pulls",
            params={"state": state, "per_page": 100},
        )

    def get_pull_request(self, pr_number: int) -> dict:
        """Get pull request details"""
        return self.request(
            "GET",
            f"{self._repo_path()}/pulls/{pr_number}",
        )

    def get_pr_files(self, pr_number: int) -> list[dict]:
        """Get pull request files"""
        return self.request(
            "GET",
            f"{self._repo_path()}/pulls/{pr_number}/files",
        )

    def get_pr_files_json(self, pr_number: int) -> list[dict]:
        """Get pull request files from the documented files.json endpoint."""
        return self.request(
            "GET",
            f"{self._repo_path()}/pulls/{pr_number}/files.json",
        )

    def get_pr_commits(self, pr_number: int) -> list[dict]:
        """Get pull request commits"""
        return self.request(
            "GET",
            f"{self._repo_path()}/pulls/{pr_number}/commits",
        )

    def get_pr_comments(self, pr_number: int) -> list[dict]:
        """Get pull request comments"""
        return self.request(
            "GET",
            f"{self._repo_path()}/pulls/{pr_number}/comments",
        )

    def get_pr_comments_page(self, pr_number: int, page: int = 1, per_page: int = 100) -> list[dict]:
        """Get one paginated page of pull request comments."""
        return self.request(
            "GET",
            f"{self._repo_path()}/pulls/{pr_number}/comments",
            params={"page": page, "per_page": per_page},
        )

    def get_all_pr_comments(self, pr_number: int, per_page: int = 100, max_pages: int = 1000) -> list[dict]:
        """Get all pull request comments across paginated API results."""
        if max_pages <= 0:
            raise ValueError("max_pages must be greater than zero")

        comments: list[dict] = []
        seen_keys: set[int | str] = set()
        page = 1

        while page <= max_pages:
            page_comments = self.get_pr_comments_page(pr_number, page=page, per_page=per_page)
            if not page_comments:
                break

            for comment in page_comments:
                comment_id = comment.get("id")
                # AtomGit may occasionally omit the integer id; fall back to a stable
                # fingerprint so repeated pagination payloads still dedupe cleanly.
                dedup_key: int | str
                if isinstance(comment_id, int):
                    dedup_key = comment_id
                else:
                    dedup_key = json.dumps(comment, sort_keys=True, ensure_ascii=False)

                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)
                comments.append(comment)

            if len(page_comments) < per_page:
                break
            page += 1
        else:
            raise AtomGitAPIError(
                f"Exceeded max_pages={max_pages} while listing PR comments for pull request {pr_number}"
            )

        return comments

    def get_current_user(self) -> dict:
        """Get the authenticated user profile."""
        return self.request("GET", "/api/v5/user")

    def get_pr_comment(self, comment_id: int) -> dict:
        """Get one pull request review comment."""
        return self.request("GET", f"{self._repo_path()}/pulls/comments/{comment_id}")

    def edit_pr_comment(self, comment_id: int, body: str) -> dict:
        """Edit one pull request review comment."""
        return self.request(
            "PATCH",
            f"{self._repo_path()}/pulls/comments/{comment_id}",
            body={"body": body},
        )

    def delete_pr_comment(self, comment_id: int) -> dict:
        """Delete one pull request review comment."""
        return self.request("DELETE", f"{self._repo_path()}/pulls/comments/{comment_id}")

    def reply_to_pr_discussion(self, pr_number: int, discussion_id: str, body: str) -> dict:
        """Reply to a specific PR review discussion."""
        return self.request(
            "POST",
            f"{self._repo_path()}/pulls/{pr_number}/discussions/{discussion_id}/comments",
            body={"body": body},
        )

    def set_pr_discussion_resolved(self, pr_number: int, discussion_id: str, resolved: bool = True) -> dict:
        """Set the resolved state for a PR review discussion."""
        return self.request(
            "PUT",
            f"{self._repo_path()}/pulls/{pr_number}/comments/{discussion_id}",
            body={"resolved": resolved},
        )

    def get_pr_comment_modify_history(self, comment_id: int) -> list[dict]:
        """Get modify history for one PR review comment."""
        return self.request(
            "GET",
            f"{self._repo_path()}/pulls/comment/{comment_id}/modify_history",
        )

    def get_pr_comment_reactions(self, comment_id: int) -> list[dict]:
        """Get reactions for one PR review comment."""
        return self.request(
            "GET",
            f"{self._repo_path()}/pulls/comment/{comment_id}/user_reactions",
        )

    def get_pr_diff(self, pr_number: int) -> dict[str, dict]:
        """Get pull request diff"""
        files = self.get_pr_files(pr_number)
        diffs = {}

        for file in files:
            if file.get("patch"):
                patch_content = (
                    file["patch"].get("diff", file["patch"]) if isinstance(file["patch"], dict) else file["patch"]
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
            f"{self._repo_path()}/contents/{encoded_path}",
            params={"ref": ref},
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
        elif comment.get("new_line") is not None:
            payload["new_line"] = comment["new_line"]
        elif comment.get("old_line") is not None:
            payload["old_line"] = comment["old_line"]
        elif comment.get("line"):
            payload["new_line"] = comment["line"]
        else:
            raise AtomGitAPIError(f"Cannot submit inline comment for {comment['path']}: no position or line provided")
        if comment.get("commit_id"):
            payload["commit_id"] = comment["commit_id"]
        elif comment.get("commitId"):
            payload["commit_id"] = comment["commitId"]

        return self.request(
            "POST",
            f"{self._repo_path()}/pulls/{pr_number}/comments",
            body=payload,
        )

    def submit_pr_comment(self, pr_number: int, body: str) -> dict:
        """Submit PR-level comment"""
        return self.request(
            "POST",
            f"{self._repo_path()}/pulls/{pr_number}/comments",
            body={"body": body},
        )

    def submit_batch_comments(self, pr_number: int, comments: list[dict]) -> list[dict]:
        """Submit batch comments"""
        results = []
        comment_base_url = f"https://atomgit.com/{self.config.owner}/{self.config.repo}/pulls/{pr_number}"

        for comment in comments:
            try:
                result = self.submit_inline_comment(pr_number, comment)
                comment_url = (
                    f"{comment_base_url}#comment-{result.get('id', '')}" if result.get("id") else comment_base_url
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
            raise AtomGitAPIError("Creating PR requires title, head, and base parameters")

        final_head = head if ":" in head else f"{self.config.owner}:{head}"

        return self.request(
            "POST",
            f"{self._repo_path()}/pulls",
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
        title: str | None = None,
        body: str | None = None,
        state: str | None = None,
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
            f"{self._repo_path()}/pulls/{pr_number}",
            body=payload,
        )

    def get_pr_url(self, pr_number: int) -> str:
        """Get PR URL"""
        return f"https://atomgit.com/{self.config.owner}/{self.config.repo}/pull/{pr_number}"

    def get_issues(self, state: str = "open") -> list[dict]:
        """Get list of issues"""
        return self.request(
            "GET",
            f"{self._repo_path()}/issues",
            params={"state": state, "per_page": 100},
        )

    def get_issue(self, issue_number: int) -> dict:
        """Get issue details"""
        return self.request(
            "GET",
            f"{self._repo_path()}/issues/{issue_number}",
        )

    def get_issue_comments(self, issue_number: int) -> list[dict]:
        """Get issue comments"""
        return self.request(
            "GET",
            f"{self._repo_path()}/issues/{issue_number}/comments",
        )

    def submit_issue_comment(self, issue_number: int, body: str) -> dict:
        """Submit an issue comment."""
        return self.request(
            "POST",
            f"{self._repo_path()}/issues/{issue_number}/comments",
            body={"body": body},
        )

    def get_issue_comment(self, comment_id: int) -> dict:
        """Get one issue comment."""
        return self.request("GET", f"{self._repo_path()}/issues/comments/{comment_id}")

    def edit_issue_comment(self, comment_id: int, body: str) -> dict:
        """Edit one issue comment."""
        return self.request(
            "PATCH",
            f"{self._repo_path()}/issues/comments/{comment_id}",
            body={"body": body},
        )

    def delete_issue_comment(self, comment_id: int) -> dict:
        """Delete one issue comment."""
        return self.request("DELETE", f"{self._repo_path()}/issues/comments/{comment_id}")

    def get_labels(self) -> list[dict]:
        """Get repository labels"""
        return self.request(
            "GET",
            f"{self._repo_path()}/labels",
        )

    def _validate_labels_exist(self, labels: list[str]) -> None:
        """Ensure all requested labels already exist in the target repository."""
        if isinstance(labels, str):
            requested_labels = [label.strip() for label in labels.split(",") if label.strip()]
        else:
            requested_labels = labels

        existing_labels = {
            label.get("name") for label in self.get_labels() if isinstance(label, dict) and label.get("name")
        }
        missing_labels = [label for label in requested_labels if label not in existing_labels]
        if missing_labels:
            raise AtomGitAPIError("Unknown labels for target repository: " + ", ".join(sorted(missing_labels)))

    def create_issue(
        self,
        title: str,
        body: str = "",
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
    ) -> dict:
        """Create issue"""
        payload = {"title": title, "body": body}
        if labels:
            self._validate_labels_exist(labels)
            payload["labels"] = ",".join(labels) if isinstance(labels, list) else labels
        if assignees:
            payload["assignees"] = ",".join(assignees) if isinstance(assignees, list) else assignees

        return self.request(
            "POST",
            f"{self._repo_path()}/issues",
            body=payload,
        )

    def update_issue(
        self,
        issue_number: int,
        title: str | None = None,
        body: str | None = None,
        state: str | None = None,
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
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
            payload["assignees"] = ",".join(assignees) if isinstance(assignees, list) else assignees

        return self.request(
            "PATCH",
            f"{self._repo_path()}/issues/{issue_number}",
            body=payload,
        )

    def get_issue_url(self, issue_number: int) -> str:
        """Get issue URL"""
        return f"https://atomgit.com/{self.config.owner}/{self.config.repo}/issues/{issue_number}"

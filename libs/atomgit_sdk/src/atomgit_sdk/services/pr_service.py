"""
PR Service - High-level PR operations
"""

import json
import warnings

from atomgit_sdk import ArchitectureIssue, BaseIssue, CodeIssue
from atomgit_sdk.client import AtomGitClient
from atomgit_sdk.utils.content import add_line_numbers
from atomgit_sdk.utils.diff import calculate_diff_position


class PRService:
    """High-level PR operations service"""

    def __init__(self, client: AtomGitClient):
        self.client = client

    def get_pr(self, pr_number: int) -> dict:
        """Get PR details"""
        return self.client.get_pull_request(pr_number)

    def get_pr_files(self, pr_number: int) -> list[dict]:
        """Get PR files"""
        return self.client.get_pr_files(pr_number)

    def get_pr_commits(self, pr_number: int) -> list[dict]:
        """Get PR commits"""
        return self.client.get_pr_commits(pr_number)

    def get_pr_diff(self, pr_number: int) -> dict[str, dict]:
        """Get PR diff"""
        return self.client.get_pr_diff(pr_number)

    def get_pr_comments(self, pr_number: int) -> list[dict]:
        """Get PR comments"""
        return self.client.get_pr_comments(pr_number)

    def get_pr_comment(self, comment_id: int) -> dict:
        """Get a single PR review comment"""
        return self.client.get_pr_comment(comment_id)

    def reply_to_pr_discussion(self, pr_number: int, discussion_id: str, body: str) -> dict:
        """Reply to a specific PR review discussion"""
        return self.client.reply_to_pr_discussion(pr_number, discussion_id, body)

    def set_pr_discussion_resolved(self, pr_number: int, discussion_id: str, resolved: bool = True) -> dict:
        """Resolve or reopen a PR review discussion"""
        return self.client.set_pr_discussion_resolved(pr_number, discussion_id, resolved)

    def get_full_pr_context(self, pr_number: int) -> dict:
        """
        Get full PR context including details, files, commits, and diffs.

        Returns:
            Dictionary with pr, files, commits, and diffs keys
        """
        pr = self.get_pr(pr_number)
        files = self.get_pr_files(pr_number)
        commits = self.get_pr_commits(pr_number)
        diffs = self.get_pr_diff(pr_number)
        comments = self.get_pr_comments(pr_number)

        return {
            "pr": pr,
            "files": files,
            "commits": commits,
            "diffs": diffs,
            "comments": comments,
        }

    def extract_pr_info(self, pr_number: int) -> dict:
        """
        Extract PR information for code review.

        Args:
            pr_number: PR number

        Returns:
            Dictionary with PR details and changed files
        """
        pr = self.client.get_pull_request(pr_number)
        files = self.client.get_pr_files(pr_number)
        head_sha = pr.get("head", {}).get("sha", "HEAD")

        changed_files = []
        for f in files:
            if f.get("status") != "removed":
                file_data = {
                    "filename": f.get("filename"),
                    "status": f.get("status"),
                    "additions": f.get("additions", 0),
                    "deletions": f.get("deletions", 0),
                    "patch": f.get("patch"),
                }

                try:
                    content = self.client.get_file_content(f.get("filename"), head_sha)
                    file_data["content"] = add_line_numbers(content)
                except Exception as e:
                    file_data["content"] = f"# Error fetching content: {e}"

                changed_files.append(file_data)

        return {
            "pr": {
                "number": pr.get("number"),
                "title": pr.get("title"),
                "author": pr.get("user", {}).get("login"),
                "branch": f"{pr.get('head', {}).get('ref')} → {pr.get('base', {}).get('ref')}",
                "head_sha": head_sha,
                "changed_files": changed_files,
            }
        }

    def load_issues_from_json(self, json_path: str) -> list[BaseIssue]:
        """
        Load issues from JSON file with deprecation warnings for camelCase fields.

        Args:
            json_path: Path to JSON file

        Returns:
            List of CodeIssue or ArchitectureIssue instances

        Raises:
            FileNotFoundError: If JSON file doesn't exist
            ValueError: If JSON is invalid
        """
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        issues = []
        for item in data:
            # Check for deprecated camelCase fields
            deprecated_fields = {
                "contextCode": "context_code",
                "fixCode": "fix_code",
                "fixExplanation": "fix_explanation",
            }

            for old_field, new_field in deprecated_fields.items():
                if old_field in item:
                    warnings.warn(
                        f"Field '{old_field}' is deprecated, use '{new_field}' instead. "
                        f"Support will be removed in version 0.3.0",
                        DeprecationWarning,
                        stacklevel=2,
                    )

            # Determine issue type
            if "pillar" in item:
                issue = ArchitectureIssue(
                    file=item.get("file", ""),
                    line=item.get("line", 0),
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    severity=item.get("severity", "warning"),
                    pillar=item.get("pillar", "python"),
                    context_code=item.get("context_code") or item.get("contextCode"),
                    fix=item.get("fix"),
                )
            else:
                issue = CodeIssue(
                    file=item.get("file", ""),
                    line=item.get("line", 0),
                    type=item.get("type", "bug"),
                    severity=item.get("severity", "warning"),
                    confidence=item.get("confidence", 80),
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    context_code=item.get("context_code") or item.get("contextCode"),
                    fix_code=item.get("fix_code") or item.get("fixCode"),
                    fix_explanation=item.get("fix_explanation") or item.get("fixExplanation"),
                )

            issues.append(issue)

        return issues

    def submit_issues(
        self,
        pr_number: int,
        issues: list[BaseIssue],
        confidence_threshold: int = 80,
        deduplicate: bool = True,
    ) -> dict:
        """
        Submit issues to PR with automatic diff position calculation.

        Args:
            pr_number: PR number
            issues: List of issues to submit
            confidence_threshold: Minimum confidence for CodeIssues (default: 80)
            deduplicate: Whether to deduplicate issues (default: True)

        Returns:
            Dictionary with submission results
        """
        # Filter by confidence for CodeIssues
        filtered_issues = [
            issue for issue in issues if not isinstance(issue, CodeIssue) or issue.confidence >= confidence_threshold
        ]

        # Deduplicate if requested
        if deduplicate:
            seen = set()
            unique_issues = []
            for issue in filtered_issues:
                key = (issue.file, issue.line, issue.title)
                if key not in seen:
                    seen.add(key)
                    unique_issues.append(issue)
            filtered_issues = unique_issues

        # Calculate positions for all issues
        diffs = self.client.get_pr_diff(pr_number)
        positions = {}

        for issue in filtered_issues:
            if issue.file not in positions:
                positions[issue.file] = {}

            diff_info = diffs.get(issue.file, {})
            is_new_file = diff_info.get("status") == "added"
            patch = diff_info.get("patch", "")
            position = calculate_diff_position(patch, issue.line, is_new_file)

            if position is not None:
                positions[issue.file][issue.line] = position

        # Prepare comments
        comments = []
        for issue in filtered_issues:
            file_positions = positions.get(issue.file, {})
            position = file_positions.get(issue.line)

            comment = {
                "path": issue.file,
                "body": self._format_issue_body(issue),
            }

            if position is not None:
                comment["position"] = position
            else:
                comment["new_line"] = issue.line

            if hasattr(issue, "commit_id") and issue.commit_id:
                comment["commit_id"] = issue.commit_id

            comments.append(comment)

        # Submit summary comment
        summary = self._format_summary(filtered_issues, pr_number)
        self.client.submit_pr_comment(pr_number, summary)

        # Submit inline comments
        if comments:
            results = self.client.submit_batch_comments(pr_number, comments)
            success_count = sum(1 for r in results if r["success"])
        else:
            results = []
            success_count = 0

        return {
            "total_issues": len(issues),
            "filtered_issues": len(filtered_issues),
            "submitted_comments": success_count,
            "results": results,
        }

    def _format_issue_body(self, issue: BaseIssue) -> str:
        """Format issue as comment body"""
        body = f"**{issue.title}**\n\n{issue.description}\n\n"

        if isinstance(issue, CodeIssue) and issue.fix_code:
            body += f"**Suggested fix:**\n```\n{issue.fix_code}\n```\n\n"
            if issue.fix_explanation:
                body += f"{issue.fix_explanation}\n\n"

        if isinstance(issue, ArchitectureIssue) and issue.fix:
            body += f"**Fix:** {issue.fix}\n\n"

        return body

    def _format_summary(self, issues: list[BaseIssue], pr_number: int) -> str:
        """Format issues summary"""
        if not issues:
            return f"## Code Review Complete\n\n✅ No issues found for PR #{pr_number}.\n\n---\n🤖 Generated by AtomGit SDK"

        body = f"## Code Review Summary for PR #{pr_number}\n\n"
        body += f"Found **{len(issues)}** issue(s):\n\n"

        for issue in issues:
            severity_icon = {
                "critical": "🚨",
                "error": "❌",
                "warning": "⚠️",
                "suggestion": "💡",
            }.get(issue.severity, "📝")

            body += f"- {severity_icon} `{issue.file}:{issue.line}` - {issue.title}\n"

        body += "\n---\n🤖 Generated by AtomGit SDK"

        return body

    def get_file_content(self, file_path: str, ref: str = "HEAD") -> str:
        """Get file content"""
        return self.client.get_file_content(file_path, ref)

    def submit_inline_comment(
        self,
        pr_number: int,
        file_path: str,
        line_number: int,
        body: str,
        commit_id: str | None = None,
        diffs: dict[str, dict] | None = None,
    ) -> dict:
        """
        Submit inline comment to PR.

        Args:
            pr_number: PR number
            file_path: File path
            line_number: Line number in the file (1-indexed)
            body: Comment body
            commit_id: Optional commit ID
            diffs: Pre-fetched diffs (optional, improves performance)

        Returns:
            API response
        """
        # Performance optimization: Use pre-fetched diffs if provided
        if diffs is None:
            diffs = self.get_pr_diff(pr_number)

        if file_path not in diffs:
            raise ValueError(f"File {file_path} not found in PR {pr_number}")

        patch = diffs[file_path]["patch"]
        position = calculate_diff_position(
            patch,
            line_number,
            is_new_file=(diffs[file_path]["status"] == "added"),
        )

        comment = {
            "path": file_path,
            "body": body,
        }

        # API Logic Fix: Properly distinguish file_line and patch_position
        if position is not None:
            comment["position"] = position
        else:
            # Position calculation failed, use new_line
            comment["new_line"] = line_number

        if commit_id:
            comment["commit_id"] = commit_id

        return self.client.submit_inline_comment(pr_number, comment)

    def submit_batch_comments(self, pr_number: int, comments: list[dict]) -> list[dict]:
        """
        Submit batch comments to PR.

        Args:
            pr_number: PR number
            comments: List of comment dictionaries with path, line, and body keys

        Returns:
            List of results with success status
        """
        diffs = self.get_pr_diff(pr_number)

        enriched_comments = []
        for comment in comments:
            file_path = comment.get("path")
            line_number = comment.get("line")

            if file_path in diffs:
                patch = diffs[file_path]["patch"]
                position = calculate_diff_position(
                    patch,
                    line_number,
                    is_new_file=(diffs[file_path]["status"] == "added"),
                )

                enriched_comment = {
                    "path": file_path,
                    "body": comment.get("body", ""),
                    "line": line_number,
                }

                # API Logic Fix: Properly distinguish file_line and patch_position
                if position is not None:
                    # Use calculated position (relative position in diff)
                    enriched_comment["position"] = position
                else:
                    # Position calculation failed, use new_line (actual file line number)
                    # AtomGit v5 API supports new_line field
                    enriched_comment["new_line"] = line_number

                if comment.get("commitId"):
                    enriched_comment["commit_id"] = comment["commitId"]

                enriched_comments.append(enriched_comment)

        return self.client.submit_batch_comments(pr_number, enriched_comments)

    def submit_pr_comment(self, pr_number: int, body: str) -> dict:
        """Submit PR-level comment"""
        return self.client.submit_pr_comment(pr_number, body)

    def create_pr(
        self,
        title: str,
        body: str,
        head: str,
        base: str = "master",
        draft: bool = False,
    ) -> dict:
        """Create PR"""
        return self.client.create_pull_request(title, body, head, base, draft)

    def update_pr(
        self,
        pr_number: int,
        title: str | None = None,
        body: str | None = None,
        state: str | None = None,
    ) -> dict:
        """Update PR"""
        return self.client.update_pull_request(pr_number, title, body, state)

    def get_pr_url(self, pr_number: int) -> str:
        """Get PR URL"""
        return self.client.get_pr_url(pr_number)

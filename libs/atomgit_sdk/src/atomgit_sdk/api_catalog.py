"""AtomGit API endpoint catalog utilities.

The public AtomGit/GitCode API documentation currently exposes most pages as a
title, HTTP method, and path.  This module keeps those docs useful in the SDK by
making documented routes discoverable and callable without adding hundreds of
thin hand-written wrappers.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from html import unescape
from urllib.request import urlopen

from atomgit_sdk.exceptions import ConfigurationError

API_DOCS_BASE_URL = "https://docs.atomgit.com/docs/apis"
API_DOCS_SITEMAP_URL = "https://docs.atomgit.com/sitemap.xml"
_PATH_PARAM_RE = re.compile(r":([A-Za-z_][A-Za-z0-9_]*)")


@dataclass(frozen=True)
class APIEndpoint:
    """One documented AtomGit API endpoint."""

    slug: str
    method: str
    path: str
    title: str = ""
    doc_url: str = ""

    @property
    def path_params(self) -> list[str]:
        """Return path parameter names required by this endpoint."""
        return _PATH_PARAM_RE.findall(self.path)

    def render(self, path_params: Mapping[str, object] | None = None) -> str:
        """Render this endpoint path by replacing ``:name`` placeholders."""
        values = dict(path_params or {})
        missing = [name for name in self.path_params if name not in values]
        if missing:
            raise ConfigurationError(f"Missing path parameters for {self.slug}: {', '.join(missing)}")

        rendered = self.path
        for name, value in values.items():
            rendered = rendered.replace(f":{name}", str(value))
        return rendered


class APICatalog:
    """Searchable collection of documented AtomGit API endpoints."""

    def __init__(self, endpoints: Iterable[APIEndpoint]):
        self._endpoints = tuple(endpoints)
        self._by_slug: dict[str, APIEndpoint] = {endpoint.slug: endpoint for endpoint in self._endpoints}

    def __len__(self) -> int:
        return len(self._endpoints)

    def all(self) -> Sequence[APIEndpoint]:
        """Return all endpoints in this catalog."""
        return self._endpoints

    def get(self, slug: str) -> APIEndpoint:
        """Get an endpoint by documentation slug."""
        try:
            return self._by_slug[slug]
        except KeyError as exc:
            raise ConfigurationError(f"Unknown AtomGit API endpoint: {slug}") from exc

    def find(
        self,
        *,
        method: str | None = None,
        path_contains: str | None = None,
        title_contains: str | None = None,
    ) -> list[APIEndpoint]:
        """Find endpoints by method, path substring, or title substring."""
        method = method.upper() if method else None
        results = list(self._endpoints)
        if method:
            results = [endpoint for endpoint in results if endpoint.method == method]
        if path_contains:
            results = [endpoint for endpoint in results if path_contains in endpoint.path]
        if title_contains:
            results = [endpoint for endpoint in results if title_contains in endpoint.title]
        return results

    @classmethod
    def from_entries(cls, entries: Iterable[tuple[str, str, str, str]]) -> APICatalog:
        """Build a catalog from ``(slug, method, path, title)`` entries."""
        return cls(
            APIEndpoint(
                slug=slug,
                method=method.upper(),
                path=path,
                title=title,
                doc_url=f"{API_DOCS_BASE_URL}/{slug}",
            )
            for slug, method, path, title in entries
        )

    @classmethod
    def from_docs(
        cls,
        sitemap_url: str = API_DOCS_SITEMAP_URL,
        *,
        timeout: int = 20,
    ) -> APICatalog:
        """Load the full endpoint catalog from the official AtomGit docs."""
        sitemap = urlopen(sitemap_url, timeout=timeout).read().decode("utf-8")
        slugs = re.findall(r"https://docs\.gitcode\.com/docs/apis/([^<]+)", sitemap)
        api_slugs = [slug for slug in slugs if re.match(r"^(get|post|put|patch|delete)-", slug)]

        endpoints = []
        for slug in api_slugs:
            page_url = f"{API_DOCS_BASE_URL}/{slug}"
            page = urlopen(page_url, timeout=timeout).read().decode("utf-8", "replace")
            endpoint = _parse_endpoint_page(slug, page, page_url)
            endpoints.append(endpoint)
        return cls(endpoints)


def _parse_endpoint_page(slug: str, html: str, page_url: str) -> APIEndpoint:
    text = re.sub(r"<[^>]+>", "\n", html)
    lines = [unescape(line).strip() for line in text.splitlines() if line.strip()]

    method = None
    path = None
    title = ""
    for index, line in enumerate(lines):
        if line in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            method = line
            title = lines[index - 1] if index > 0 else ""
            for candidate in lines[index + 1 : index + 8]:
                if candidate.startswith("/"):
                    path = candidate.replace("\\_", "_")
                    break
            break

    if not method or not path:
        raise ConfigurationError(f"Could not parse AtomGit API doc page: {page_url}")

    return APIEndpoint(slug=slug, method=method, path=path, title=title, doc_url=page_url)


# Built-in catalog for the SDK-supported collaboration surface.  Use
# APICatalog.from_docs() when a workflow needs the full official API catalog.
DEFAULT_API_CATALOG = APICatalog.from_entries(
    (
        ("get-api-v-5-user", "GET", "/api/v5/user", "获取授权用户的资料"),
        ("get-api-v-5-users-username", "GET", "/api/v5/users/:username", "获取一个用户"),
        ("get-api-v-5-repos-owner-repo", "GET", "/api/v5/repos/:owner/:repo", "获取一个仓库"),
        ("patch-api-v-5-repos-owner-repo", "PATCH", "/api/v5/repos/:owner/:repo", "更新仓库设置"),
        ("delete-api-v-5-repos-owner-repo", "DELETE", "/api/v5/repos/:owner/:repo", "删除一个仓库"),
        (
            "get-api-v-5-repos-owner-repo-contents-path",
            "GET",
            "/api/v5/repos/:owner/:repo/contents/:path",
            "获取仓库具体路径下的内容",
        ),
        (
            "post-api-v-5-repos-owner-repo-contents-path",
            "POST",
            "/api/v5/repos/:owner/:repo/contents/:path",
            "新建文件",
        ),
        ("put-api-v-5-repos-owner-repo-contents-path", "PUT", "/api/v5/repos/:owner/:repo/contents/:path", "更新文件"),
        (
            "delete-api-v-5-repos-owner-repo-contents-path",
            "DELETE",
            "/api/v5/repos/:owner/:repo/contents/:path",
            "删除文件",
        ),
        ("get-api-v-5-repos-owner-repo-branches", "GET", "/api/v5/repos/:owner/:repo/branches", "获取仓库分支列表"),
        (
            "get-api-v-5-repos-owner-repo-branches-branch",
            "GET",
            "/api/v5/repos/:owner/:repo/branches/:branch",
            "获取单个分支",
        ),
        ("post-api-v-5-repos-owner-repo-branches", "POST", "/api/v5/repos/:owner/:repo/branches", "创建分支"),
        (
            "delete-api-v-5-repos-owner-repo-branches-name",
            "DELETE",
            "/api/v5/repos/:owner/:repo/branches/:name",
            "删除分支",
        ),
        ("get-api-v-5-repos-owner-repo-commits", "GET", "/api/v5/repos/:owner/:repo/commits", "获取仓库所有提交"),
        (
            "get-api-v-5-repos-owner-repo-commits-sha",
            "GET",
            "/api/v5/repos/:owner/:repo/commits/:sha",
            "获取仓库某个提交",
        ),
        (
            "get-api-v-5-repos-owner-repo-commits-sha-diff",
            "GET",
            "/api/v5/repos/:owner/:repo/commits/:sha/diff",
            "获取提交差异",
        ),
        (
            "get-api-v-5-repos-owner-repo-compare-base-head",
            "GET",
            "/api/v5/repos/:owner/:repo/compare/:base...:head",
            "比较两个提交",
        ),
        ("get-api-v-5-repos-owner-repo-labels", "GET", "/api/v5/repos/:owner/:repo/labels", "获取仓库标签"),
        ("post-api-v-5-repos-owner-repo-labels", "POST", "/api/v5/repos/:owner/:repo/labels", "创建标签"),
        (
            "patch-api-v-5-repos-owner-repo-labels-original-name",
            "PATCH",
            "/api/v5/repos/:owner/:repo/labels/:original_name",
            "更新标签",
        ),
        (
            "delete-api-v-5-repos-owner-repo-labels-name",
            "DELETE",
            "/api/v5/repos/:owner/:repo/labels/:name",
            "删除标签",
        ),
        ("get-api-v-5-repos-owner-repo-issues", "GET", "/api/v5/repos/:owner/:repo/issues", "仓库Issue列表"),
        (
            "get-api-v-5-repos-owner-repo-issues-number",
            "GET",
            "/api/v5/repos/:owner/:repo/issues/:number",
            "获取仓库某个Issue",
        ),
        ("post-api-v-5-repos-owner-issues", "POST", "/api/v5/repos/:owner/issues", "创建Issue"),
        ("patch-api-v-5-repos-owner-issues-number", "PATCH", "/api/v5/repos/:owner/issues/:number", "更新Issue"),
        (
            "get-api-v-5-repos-owner-repo-issues-number-comments",
            "GET",
            "/api/v5/repos/:owner/:repo/issues/:number/comments",
            "获取Issue评论",
        ),
        (
            "post-api-v-5-repos-owner-repo-issues-number-comments",
            "POST",
            "/api/v5/repos/:owner/:repo/issues/:number/comments",
            "创建Issue评论",
        ),
        (
            "get-api-v-5-repos-owner-repo-issues-comments-id",
            "GET",
            "/api/v5/repos/:owner/:repo/issues/comments/:id",
            "获取Issue某条评论",
        ),
        (
            "patch-api-v-5-repos-owner-repo-issues-comments-id",
            "PATCH",
            "/api/v5/repos/:owner/:repo/issues/comments/:id",
            "编辑Issue评论",
        ),
        (
            "delete-api-v-5-repos-owner-repo-issues-comments-id",
            "DELETE",
            "/api/v5/repos/:owner/:repo/issues/comments/:id",
            "删除Issue评论",
        ),
        ("get-api-v-5-repos-owner-repo-pulls", "GET", "/api/v5/repos/:owner/:repo/pulls", "Pull Request列表"),
        ("post-api-v-5-repos-owner-repo-pulls", "POST", "/api/v5/repos/:owner/:repo/pulls", "创建Pull Request"),
        (
            "get-api-v-5-repos-owner-repo-pulls-number",
            "GET",
            "/api/v5/repos/:owner/:repo/pulls/:number",
            "获取某个Pull Request",
        ),
        (
            "patch-api-v-5-repos-owner-repo-pulls-number",
            "PATCH",
            "/api/v5/repos/:owner/:repo/pulls/:number",
            "更新Pull Request",
        ),
        (
            "get-api-v-5-repos-owner-repo-pulls-number-files",
            "GET",
            "/api/v5/repos/:owner/:repo/pulls/:number/files",
            "Pull Request文件列表",
        ),
        (
            "get-api-v-5-repos-owner-repo-pulls-number-files-json",
            "GET",
            "/api/v5/repos/:owner/:repo/pulls/:number/files.json",
            "Pull Request Commit文件列表",
        ),
        (
            "get-api-v-5-repos-owner-repo-pulls-number-commits",
            "GET",
            "/api/v5/repos/:owner/:repo/pulls/:number/commits",
            "Pull Request提交列表",
        ),
        (
            "get-api-v-5-repos-owner-repo-pulls-number-comments",
            "GET",
            "/api/v5/repos/:owner/:repo/pulls/:number/comments",
            "获取某个Pull Request的所有评论",
        ),
        (
            "post-api-v-5-repos-owner-repo-pulls-number-comments",
            "POST",
            "/api/v5/repos/:owner/:repo/pulls/:number/comments",
            "提交pull request 评论",
        ),
        (
            "get-api-v-5-repos-owner-repo-pulls-comments-id",
            "GET",
            "/api/v5/repos/:owner/:repo/pulls/comments/:id",
            "获取Pull Request某条评论",
        ),
        (
            "patch-api-v-5-repos-owner-repo-pulls-comments-id",
            "PATCH",
            "/api/v5/repos/:owner/:repo/pulls/comments/:id",
            "编辑评论",
        ),
        (
            "delete-api-v-5-repos-owner-repo-pulls-comments-id",
            "DELETE",
            "/api/v5/repos/:owner/:repo/pulls/comments/:id",
            "删除评论",
        ),
        (
            "post-api-v-5-repos-owner-repo-pulls-number-discussions-discussions-id-comments",
            "POST",
            "/api/v5/repos/:owner/:repo/pulls/:number/discussions/:discussion_id/comments",
            "回复Pull Request评论",
        ),
        (
            "put-api-v-5-repos-owner-repo-pulls-number-comments-discussions-id",
            "PUT",
            "/api/v5/repos/:owner/:repo/pulls/:number/comments/:discussion_id",
            "修改检视意见解决状态",
        ),
        (
            "post-api-v-5-repos-owner-repo-pulls-number-review",
            "POST",
            "/api/v5/repos/:owner/:repo/pulls/:number/review",
            "提交Pull Request Review",
        ),
        (
            "get-api-v-5-repos-owner-repo-pulls-comment-comment-id-modify-history",
            "GET",
            "/api/v5/repos/:owner/:repo/pulls/comment/:comment_id/modify_history",
            "获取Pull Request评论的修改历史",
        ),
        (
            "get-api-v-5-repos-owner-repo-pulls-comment-comment-id-user-reactions",
            "GET",
            "/api/v5/repos/:owner/:repo/pulls/comment/:comment_id/user_reactions",
            "获取Pull Request评论的表态列表",
        ),
        (
            "get-api-v-5-repos-owner-repo-pulls-number-user-reactions",
            "GET",
            "/api/v5/repos/:owner/:repo/pulls/:number/user_reactions",
            "获取Pull Request的表态列表",
        ),
        ("get-api-v-5-search-repositories", "GET", "/api/v5/search/repositories", "搜索仓库"),
        ("get-api-v-5-search-issues", "GET", "/api/v5/search/issues", "搜索Issue"),
        ("get-api-v-5-search-users", "GET", "/api/v5/search/users", "搜索用户"),
    )
)

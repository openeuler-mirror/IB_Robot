"""URL parsing utilities."""

from urllib.parse import urlparse
from typing import Dict, Union

from atomgit_sdk.exceptions import URLError


SUPPORTED_ATOMGIT_DOMAINS = {
    "atomgit.com",
    "www.atomgit.com",
    "gitcode.com",
    "www.gitcode.com",
}


def parse_atomgit_url(url: str) -> Dict[str, Union[str, int]]:
    """
    Parse AtomGit/GitCode URL and extract repository context.

    Supported examples:
    - https://atomgit.com/owner/repo
    - https://atomgit.com/owner/repo/pull/123
    - https://atomgit.com/owner/repo/issues/456
    - https://gitcode.com/owner/repo/merge_requests/789

    Returns:
        Dictionary with owner, repo, and optionally pr_number, issue_number, or branch
    """
    if not url:
        raise URLError("Cannot parse empty URL", url=url)

    normalized_url = url if "://" in url else f"https://{url}"
    parsed = urlparse(normalized_url)

    if parsed.netloc not in SUPPORTED_ATOMGIT_DOMAINS:
        raise URLError("Unsupported AtomGit URL domain", url=url)

    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 2:
        raise URLError("Cannot parse repository owner and name", url=url)

    owner = path_parts[0]
    repo = path_parts[1].removesuffix(".git")
    result: Dict[str, Union[str, int]] = {"owner": owner, "repo": repo}

    if len(path_parts) == 2:
        return result

    resource = path_parts[2]
    if resource in {"pull", "pulls", "merge_request", "merge_requests"}:
        if len(path_parts) >= 4 and path_parts[3].isdigit():
            result["pr_number"] = int(path_parts[3])
            return result
        raise URLError("PR URL is missing a numeric PR number", url=url)

    if resource in {"issue", "issues"}:
        if len(path_parts) >= 4 and path_parts[3].isdigit():
            result["issue_number"] = int(path_parts[3])
            return result
        raise URLError("Issue URL is missing a numeric issue number", url=url)

    if resource in {"tree", "commits", "blob", "src"} and len(path_parts) >= 4:
        result["branch"] = "/".join(path_parts[3:])
        return result

    return result

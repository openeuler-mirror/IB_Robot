import hashlib
import json
import os
import re
import subprocess
import tempfile
from urllib.parse import urlparse

VERIFIED_TREE_LABEL = "Verified tree"
VERIFICATION_MODE_LABEL = "Docker verification mode"
VERIFIED_INPUTS_LABEL = "Verified inputs"
TESTED_SOURCE_TREE_LABEL = "Tested source tree"
VERIFICATION_ENVIRONMENT_LABEL = "Docker environment"
VERIFICATION_MODE_FULL = "full"
VERIFICATION_MODE_REUSED = "reused-environment"
VERIFICATION_POLICY_VERSION = "1"
WIP_PREFIX = "[WIP]"
_VERIFICATION_HEADING_RE = re.compile(r"^##[ \t]+Docker Verification[ \t]*\r?$", re.MULTILINE | re.IGNORECASE)
_VERIFIED_TREE_RE = re.compile(
    rf"\*\*{VERIFIED_TREE_LABEL}:\*\*\s*`?([0-9a-f]{{40}})(?![0-9a-f])`?",
    re.IGNORECASE,
)
_WIP_PREFIX_RE = re.compile(r"^(?:\s*\[WIP\]\s*)+", re.IGNORECASE)
# Field lines accept an optional Markdown list prefix ("- " or "* ") so Agents
# can write them inside bullet lists without triggering duplicate-block errors.
# upsert_verification_metadata strips both forms before appending the canonical
# block, so formatting never yields two copies of the same field.
_FIELD_LINE_PREFIX = r"(?:-\s+)?"
_VERIFICATION_MODE_RE = re.compile(
    rf"{_FIELD_LINE_PREFIX}\*\*{re.escape(VERIFICATION_MODE_LABEL)}:\*\*\s*`?(full|reused-environment)`?",
    re.IGNORECASE,
)
_VERIFICATION_SHA_FIELDS = {
    VERIFIED_INPUTS_LABEL: re.compile(
        rf"{_FIELD_LINE_PREFIX}\*\*{re.escape(VERIFIED_INPUTS_LABEL)}:\*\*\s*`?([0-9a-f]{{40}})(?![0-9a-f])`?",
        re.IGNORECASE,
    ),
    TESTED_SOURCE_TREE_LABEL: re.compile(
        rf"{_FIELD_LINE_PREFIX}\*\*{re.escape(TESTED_SOURCE_TREE_LABEL)}:\*\*\s*`?([0-9a-f]{{40}})(?![0-9a-f])`?",
        re.IGNORECASE,
    ),
}
_VERIFICATION_ENVIRONMENT_RE = re.compile(
    rf"{_FIELD_LINE_PREFIX}\*\*{re.escape(VERIFICATION_ENVIRONMENT_LABEL)}:\*\*\s*`([^`\n]+)`",
    re.IGNORECASE,
)
_PACKAGE_DEPENDENCY_RE = re.compile(
    r"^[+-](?![+-])\s*</?(?:depend|build_depend|build_export_depend|buildtool_depend|buildtool_export_depend|exec_depend|test_depend|doc_depend|group_depend)\b",
    re.MULTILINE,
)
_GLOBAL_GATE_FILES = {
    "CMakeLists.txt",
    "pyproject.toml",
    "scripts/build.sh",
    "scripts/install_ros.sh",
    "scripts/setup.sh",
    "scripts/setup/python_venv.sh",
    "scripts/setup/verify_env.sh",
    "scripts/setup/lerobot_patches.sh",
    ".gitmodules",
}
_LEROBOT_PATCH_PREFIX = "third_party/patches/lerobot/"


def file_triggers_dual_docker_gate(filename: str, patch: str = "") -> bool:
    if filename in _GLOBAL_GATE_FILES:
        return True
    if filename.startswith("scripts/setup/platforms/") and filename.endswith(".sh"):
        return True
    if filename.startswith("requirements/") and filename.endswith(".txt"):
        return True
    if filename == "libs/lerobot" or filename.startswith(_LEROBOT_PATCH_PREFIX):
        return True
    if filename != "package.xml" and not filename.endswith("/package.xml"):
        return False
    # Missing patches are treated as gated so API truncation cannot bypass the check.
    return not patch or bool(_PACKAGE_DEPENDENCY_RE.search(patch))


def _normalise_patch(patch: str | dict | None) -> str:
    if isinstance(patch, dict):
        patch = patch.get("diff") or ""
    return patch or ""


def _changed_lines(patch: str) -> list[str]:
    """Extract only +/- content lines, ignoring hunk headers and context."""
    lines = []
    for line in patch.splitlines():
        if not line or line[0] not in "+-" or line.startswith(("+++", "---")):
            continue
        lines.append(line)
    return lines


def verification_input_record(filename: str, patch: str | dict | None = "") -> tuple[str, str] | None:
    """Return the setup/ dependency input represented by one changed file."""
    patch = _normalise_patch(patch)
    if filename == "package.xml" or filename.endswith("/package.xml"):
        if not patch:
            return filename, "<patch-unavailable>"
        dependency_lines = [line for line in _changed_lines(patch) if _PACKAGE_DEPENDENCY_RE.search(line)]
        return (filename, "\n".join(sorted(dependency_lines))) if dependency_lines else None
    if file_triggers_dual_docker_gate(filename, patch):
        if not patch:
            return filename, "<patch-unavailable>"
        return filename, "\n".join(_changed_lines(patch))
    return None


def compute_verification_inputs(files: list[dict]) -> str | None:
    records = []
    for file_info in files:
        filename = file_info.get("filename") or file_info.get("new_path") or ""
        record = verification_input_record(filename, file_info.get("patch") or "")
        if record is not None:
            records.append(record)
    if not records:
        return None
    payload = {
        "policy": VERIFICATION_POLICY_VERSION,
        "records": sorted(set(records)),
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded).hexdigest()


def is_wip_title(title: str) -> bool:
    return bool(_WIP_PREFIX_RE.match(title or ""))


def normalize_pr_title(title: str, stage: str) -> str:
    base_title = _WIP_PREFIX_RE.sub("", title or "").strip()
    if not base_title:
        raise ValueError("PR title must not be empty")
    if stage == "wip":
        return f"{WIP_PREFIX} {base_title}"
    if stage == "review":
        return base_title
    raise ValueError("PR stage must be 'wip' or 'review'")


def resolve_pr_stage(title: str, stage: str | None, gate_required: bool) -> tuple[str, bool]:
    if gate_required and stage is None:
        raise ValueError(
            "dual Docker verification gate requires an explicit PR stage after asking the user: use 'wip' or 'review'"
        )
    if stage is not None:
        title = normalize_pr_title(title, stage)
    return title, gate_required and not is_wip_title(title)


def extract_verified_tree(description: str) -> str | None:
    matches = [match.lower() for match in _VERIFIED_TREE_RE.findall(description or "")]
    if len(matches) != 1:
        return None
    return matches[0]


def extract_verification_metadata(description: str) -> dict | None:
    if len(_VERIFICATION_HEADING_RE.findall(description or "")) > 1:
        raise ValueError("description must contain exactly one '## Docker Verification' heading")
    mode_matches = _VERIFICATION_MODE_RE.findall(description or "")
    input_matches = _VERIFICATION_SHA_FIELDS[VERIFIED_INPUTS_LABEL].findall(description or "")
    tree_matches = _VERIFICATION_SHA_FIELDS[TESTED_SOURCE_TREE_LABEL].findall(description or "")
    environment_matches = _VERIFICATION_ENVIRONMENT_RE.findall(description or "")
    if not mode_matches and not input_matches and not tree_matches:
        legacy_tree = extract_verified_tree(description)
        return {"mode": "legacy", "tested_tree": legacy_tree} if legacy_tree else None
    if len(mode_matches) != 1 or len(input_matches) != 1 or len(tree_matches) != 1 or len(environment_matches) != 1:
        counts = {
            VERIFICATION_MODE_LABEL: len(mode_matches),
            VERIFIED_INPUTS_LABEL: len(input_matches),
            TESTED_SOURCE_TREE_LABEL: len(tree_matches),
            VERIFICATION_ENVIRONMENT_LABEL: len(environment_matches),
        }
        bad_fields = ", ".join(f"{label}={count}" for label, count in counts.items() if count != 1)
        raise ValueError(
            "Docker verification metadata requires exactly one of each field; "
            f"found: {bad_fields}. Each field must appear exactly once as "
            "'**<label>:** <value>' (a leading '- ' list prefix is also accepted)."
        )
    return {
        "mode": mode_matches[0].lower(),
        "verified_inputs": input_matches[0].lower(),
        "tested_tree": tree_matches[0].lower(),
        "environment": environment_matches[0],
    }


def format_verification_metadata(
    mode: str,
    verified_inputs: str,
    tested_tree: str,
    environment: str,
) -> str:
    if mode not in {VERIFICATION_MODE_FULL, VERIFICATION_MODE_REUSED}:
        raise ValueError(f"unsupported Docker verification mode: {mode}")
    return (
        "## Docker Verification\n\n"
        f"**{VERIFICATION_MODE_LABEL}:** `{mode}`\n"
        f"**{VERIFIED_INPUTS_LABEL}:** `{verified_inputs}`\n"
        f"**{TESTED_SOURCE_TREE_LABEL}:** `{tested_tree}`\n"
        f"**{VERIFICATION_ENVIRONMENT_LABEL}:** `{environment}`"
    )


def upsert_verification_metadata(
    description: str,
    mode: str,
    verified_inputs: str,
    tested_tree: str,
    environment: str,
) -> str:
    if len(_VERIFICATION_HEADING_RE.findall(description)) > 1:
        raise ValueError("description must contain exactly one '## Docker Verification' heading")
    line_patterns = [
        rf"^(?:-\s+)?\*\*{re.escape(VERIFICATION_MODE_LABEL)}:\*\*.*(?:\n|$)",
        rf"^(?:-\s+)?\*\*{re.escape(VERIFIED_INPUTS_LABEL)}:\*\*.*(?:\n|$)",
        rf"^(?:-\s+)?\*\*{re.escape(TESTED_SOURCE_TREE_LABEL)}:\*\*.*(?:\n|$)",
        rf"^(?:-\s+)?\*\*{re.escape(VERIFICATION_ENVIRONMENT_LABEL)}:\*\*.*(?:\n|$)",
        rf"^(?:-\s+)?\*\*{re.escape(VERIFIED_TREE_LABEL)}:\*\*.*(?:\n|$)",
    ]
    cleaned = description
    for pattern in line_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.MULTILINE)
    block = format_verification_metadata(mode, verified_inputs, tested_tree, environment)
    heading = _VERIFICATION_HEADING_RE.search(cleaned)
    if heading:
        before = cleaned[: heading.start()]
        after = cleaned[heading.end() :].lstrip("\r\n")
        return f"{before}{block}\n\n{after}".rstrip() + "\n"
    return f"{cleaned.rstrip()}\n\n{block}\n"


def validate_verification_metadata(
    description: str,
    expected_inputs: str,
    expected_tree: str,
    *,
    allow_reuse: bool = False,
    expected_environment: str | None = None,
) -> dict:
    """Validate Docker verification metadata.

    For reused-environment mode, prior-evidence existence is verified by the
    author-side ``prepare_update_verification``; the reviewer only checks that
    inputs fingerprint and environment match.
    """
    metadata = extract_verification_metadata(description)
    if metadata is None:
        raise ValueError("Docker verification metadata is missing")
    if metadata["mode"] == "legacy":
        if metadata.get("tested_tree", "").lower() != expected_tree.lower():
            raise ValueError("legacy Docker verification only matches the current source tree")
        raise ValueError("legacy Docker evidence must be migrated to full verification metadata")
    if metadata["verified_inputs"] != expected_inputs.lower():
        raise ValueError(
            f"Docker verification inputs {metadata['verified_inputs']} do not match current inputs "
            f"{expected_inputs.lower()}. The expected value is a fingerprint computed from the PR "
            "file list and patches, not the commit or tree SHA. Rerun the update so the gate reports "
            "the expected fingerprint, or reuse the fingerprint recorded by the last gate run."
        )
    if not metadata.get("environment"):
        raise ValueError("Docker verification environment is missing")
    if expected_environment is not None and metadata.get("environment") != expected_environment:
        raise ValueError(
            f"Docker environment {metadata.get('environment')} does not match current environment {expected_environment}"
        )
    if metadata["mode"] == VERIFICATION_MODE_FULL:
        if expected_tree and metadata["tested_tree"] != expected_tree.lower():
            raise ValueError(
                f"full Docker verification tested tree {metadata['tested_tree']} does not match current tree "
                f"{expected_tree.lower()}. The tested tree must equal the PR head tree; re-run both "
                "platform verifications against that tree."
            )
    elif metadata["mode"] == VERIFICATION_MODE_REUSED:
        if not allow_reuse:
            raise ValueError("reused-environment verification is only valid when prior evidence is available")
    else:
        raise ValueError(f"unsupported Docker verification mode: {metadata['mode']}")
    return metadata


def prepare_update_verification(
    description: str,
    previous_description: str,
    expected_inputs: str,
    current_tree: str,
) -> tuple[str, dict]:
    """Carry forward reusable evidence or require a fresh full verification.

    The description supplied by the Agent is treated as a *draft*: when it lacks a
    current Docker verification block but the previous PR description carries reusable
    evidence with the same inputs, the workflow automatically inserts a
    ``reused-environment`` block so reviewers see explicit provenance instead of a
    silent gap. A new ``full`` block is only required when the inputs changed or no
    reusable evidence exists.
    """
    current = extract_verification_metadata(description)
    try:
        previous = extract_verification_metadata(previous_description)
    except ValueError:
        previous = None

    if current and current["mode"] == VERIFICATION_MODE_FULL:
        if current.get("verified_inputs") != expected_inputs.lower():
            raise ValueError(
                f"full Docker verification inputs {current.get('verified_inputs')} do not match the "
                f"current PR inputs {expected_inputs.lower()}. The expected inputs fingerprint is "
                "computed from the PR file list and patches; take it from the previous gate error or "
                "ask the maintainer workflow to print it, then rerun both platform verifications."
            )
        if current.get("tested_tree") != current_tree.lower():
            raise ValueError(
                f"full Docker verification tested tree {current.get('tested_tree')} does not match the "
                f"current PR tree {current_tree.lower()}. Re-run both platform verifications against "
                "the PR head tree."
            )
        if not current.get("environment"):
            raise ValueError("full Docker verification environment is missing")
        validate_verification_metadata(description, expected_inputs, current_tree)
        return upsert_verification_metadata(
            description,
            VERIFICATION_MODE_FULL,
            expected_inputs,
            current_tree,
            current["environment"],
        ), current

    previous_reusable = bool(
        previous
        and previous.get("mode") in {VERIFICATION_MODE_FULL, VERIFICATION_MODE_REUSED}
        and previous.get("verified_inputs") == expected_inputs.lower()
        and previous.get("tested_tree")
        and previous.get("environment")
    )

    if current and current["mode"] == VERIFICATION_MODE_REUSED:
        if not previous_reusable:
            raise ValueError("reused-environment verification requires previously recorded Docker evidence")
        if current.get("tested_tree") != previous["tested_tree"]:
            raise ValueError("reused-environment tested tree must match the previously verified tree")
        if not current.get("environment"):
            raise ValueError("reused-environment verification environment is missing")
        validate_verification_metadata(
            description,
            expected_inputs,
            current_tree,
            allow_reuse=True,
        )
        return upsert_verification_metadata(
            description,
            VERIFICATION_MODE_REUSED,
            expected_inputs,
            current["tested_tree"],
            current["environment"],
        ), current

    if previous_reusable:
        mode = VERIFICATION_MODE_FULL if previous["tested_tree"] == current_tree else VERIFICATION_MODE_REUSED
        tested_tree = current_tree if mode == VERIFICATION_MODE_FULL else previous["tested_tree"]
        metadata = {
            "mode": mode,
            "verified_inputs": expected_inputs,
            "tested_tree": tested_tree,
            "environment": previous["environment"],
        }
        return upsert_verification_metadata(
            description,
            mode,
            expected_inputs,
            tested_tree,
            previous["environment"],
        ), metadata

    raise ValueError(
        "a full Docker verification is required; no reusable evidence matches the current inputs. "
        "Run both platform verifications (Ubuntu + openEuler) against the PR head tree, then set "
        f"**{VERIFIED_INPUTS_LABEL}:** to {expected_inputs.lower()} — the fingerprint of the PR file "
        "list and patches, NOT the commit or tree SHA — and **Tested source tree:** to the head tree."
    )


def _run_git(args: list[str], cwd: str) -> str:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise ValueError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def resolve_pr_head_tree(pr: dict) -> str:
    head = pr.get("head") or {}
    head_sha = (head.get("sha") or "").lower()
    head_ref = head.get("ref") or ""
    repo_url = (head.get("repo") or {}).get("html_url") or ""
    parsed_url = urlparse(repo_url)
    if len(head_sha) != 40 or not head_ref or parsed_url.scheme != "https" or not parsed_url.netloc:
        raise ValueError("PR head commit, branch, or HTTPS repository URL is unavailable")

    with tempfile.TemporaryDirectory(prefix="ibrobot-pr-head-") as git_dir:
        _run_git(["init", "--bare"], git_dir)
        _run_git(["fetch", "--no-tags", "--depth=1", repo_url, f"refs/heads/{head_ref}"], git_dir)
        fetched_sha = _run_git(["rev-parse", "FETCH_HEAD"], git_dir).lower()
        if fetched_sha != head_sha:
            raise ValueError(f"PR head changed during tree resolution: API={head_sha}, fetched={fetched_sha}")
        return _run_git(["rev-parse", "FETCH_HEAD^{tree}"], git_dir).lower()

---
name: ibrobot-git-flow
description: "Handles Git commit and push workflow with Euler compliance. Use when user asks to 'git commit', 'git push', '提交代码', '推送代码', 'git status', 'commit -s', 'DCO sign-off', 'check git message', 'euler compliance', '符合规范'. Triggers for 'push to fork', 'commit changes', '提交', '检查提交信息', '整理 PR commits', 'fixup', 'autosquash', 'squash', 'PR commit太多', or when preparing code for submission. For AtomGit review comment resolution (replying to comments, applying review fixes via the resolution script), use atomgit-review-resolution instead."
---

# IB_Robot Git Workflow Guide

This skill automates code commit process for IB_Robot project (root repo with src directory) and its submodule `libs/lerobot`. The commit process enforces openEuler Embedded specification.

## Core Specifications

### 0. Commit Scope — Only Commit What You Changed

**Do NOT blindly `git add .` and commit everything.** Before staging, carefully inspect `git diff` and `git status` to ensure only files related to the current task are included.

**`libs/lerobot` exclusion rule:**
- `libs/lerobot` is managed via a **patch stack** (`third_party/patches/lerobot/`). Changes inside `libs/lerobot` must be exported as patches using the `ibrobot-lerobot-patch` skill — they should **never** be committed directly to the root repo via `git add libs/lerobot`.
- Therefore, during normal `git commit` workflow, **always exclude `libs/lerobot` from staging**. Use `git add` on specific files or paths, NOT `git add .` from the repo root.
- The only exception: the user explicitly asks to commit lerobot changes directly (e.g., "把 lerobot 的改动也提交了").

**Practical rule:**
1. Review `git status` output carefully.
2. Stage only the files that belong to the current task using `git add <specific-paths>`.
3. If `libs/lerobot` appears in `git status`, explicitly skip it with `git restore --staged libs/lerobot` or simply do not stage it.
4. If unrelated files (not part of the current task) appear modified, skip them too unless the user asks to include them.

### 1. Commit Message Format
Must strictly follow this structure with exactly one blank line between sections:

```
<area>: <subject>

<body>

<footer_tags>
```

- **Title (<area>: <subject>)**:
  - Format: `<module>: <brief description>` (e.g., `robot_interface: fix moveit crash`).
  - Length limit: Non-revert commits max 80 chars, revert commits max 102 chars.
  - Subject must have at least 2 words, no trailing punctuation.
  - Exactly one space after colon.
  - No Chinese characters allowed.
- **Body**:
  - Must provide detailed description explaining "why" and "what".
  - Each line max 100 characters (unless containing URL).
  - No Chinese characters allowed.
- **Footer (Tags)**:
  - Must include `Signed-off-by: Name <email>`.
  - `Signed-off-by` must be the last line.
  - Allowed tags: `Fixes`, `Closes`, `Co-developed-by`, `Co-Authored-By`, `Link`.
  - For AI-assisted contributions, add `Co-Authored-By: <AI model name and version>` immediately before
    `Signed-off-by`. The PR's `模型信息 (Model)` disclosure must include that model. Different commits in one PR
    may record different models, and human co-authors use `Co-Authored-By: Name <email>`.
  - Record only the model name and version (for example, `gpt-5.6-sol`). Strip runtime provider prefixes such as
    `xunxing/`; provider routing is not contribution metadata.
  - Tags must start with capital letter, one space after colon.
  - `Fixes` format: `Fixes: <12-char-SHA1>(<original-commit-title>)`.

### 2. Mandatory Requirements
- **All commits must be signed**: Use `git commit -s` to auto-add `Signed-off-by` to footer. Ensure this line is last.
- **Sign-off provenance — never fabricate identities**: The `Signed-off-by` identity must come exclusively from
  `git commit -s`, which reads the committer's `git config user.name` / `git config user.email`.
  - Before committing, confirm the identity is configured: `git config user.name && git config user.email`.
  - **NEVER hand-write, guess, or copy a `Signed-off-by: Name <email>` line** into a `-m` / `-F` message body
    or a message file. A hand-written line is exactly how fabricated foreign sign-offs sneak into history
    (e.g., signing a commit as a colleague who never touched it).
  - **NEVER add a `Signed-off-by` for another person** unless that person explicitly signed off on the change
    (DCO). When the intended author differs from `git config` identity, ask the user how to proceed instead of
    inventing a trailer.
  - After committing, verify trailers: every `Signed-off-by` must be attributable (`git commit -s` output or an
    explicit DCO confirmation). Treat an unexplained extra or mismatched sign-off as an error to rewrite
    **before push**, not after reviewers see it.
- **Disclose AI participation**: When an Agent generated or automatically processed any staged code, documentation,
  configuration, test, or script, record the actual model name and version in the commit. Never use placeholders such
  as `AI`, `agent`, or an unversioned product family, and never include a `provider/` prefix. Use this footer order:
  ```text
  Co-Authored-By: <AI model name and version>
  Signed-off-by: Name <email>
  ```
- **Keep PR metadata complete**: Before push or PR creation, collect the AI model from every AI-assisted commit and
  ensure the PR disclosure lists the complete model set. Different commits may use different models; a model present
  in a commit but absent from the PR disclosure is a blocking error. Purely human commits need no AI trailer.
  The model set is defined over **all commits of the PR**: when pushing new commits to a branch that already has an
  open PR — including commits authored in another session or by another agent — merge the new commit models into the
  disclosure. After the push, immediately refresh the PR description via `pr_management.py --update-pr`; the script
  re-validates coverage against the remote PR commits and rejects any missing model.
- **Verify Agent tool provenance**: Before invoking the PR workflow, the coding agent must run the actual tool's
  `--version` (or equivalent) command and pass the observed tool name/version as `--agent-tool`. The repository
  must not maintain an exhaustive tool allowlist or execute arbitrary unknown tools; the workflow only validates
  that the supplied provenance is concrete, versioned, and safe to include in Markdown.
- **Human responsibility and provenance**: The user's explicit request to commit, push, or submit a PR in the current
  conversation counts as the human review confirmation of the AI-assisted changes; third-party materials and licenses
  are taken from what the user provides. Do NOT block the flow with interactive ask-user prompts for confirmation.
  Still refuse clearly unexplained, sensitive, confidential, or license-incompatible content. Follow the openEuler
  policy at <https://www.openeuler.openatom.cn/zh/community/ai-coding-assistants/>.
- **Remote repositories**:
  - `origin`: Personal fork (for pushing code).
  - `upstream`: Main project repo (for submitting Pull Requests).

### 3. PR Commit Hygiene — Keep PRs Reviewable

**A single PR should contain at most 5 commits.** Reviewers must be able to read the change set quickly without walking a long commit chain.

**Never create commits that exist only to address review feedback.** Do NOT push commits with messages like:
- `fix: address review comments`
- `fix reviewer's意见`
- `apply review suggestions`
- `update based on PR feedback`

These add noise to git history, bloat the PR, and make the change set harder to follow.

**Correct way to apply review fixes — `git commit --fixup` + `git rebase --autosquash`:**

1. Before changing history, fetch the source branch and save the exact remote OID that was reviewed:
   ```bash
   BRANCH="$(git branch --show-current)"
   test -n "$BRANCH"
   git fetch origin "+refs/heads/${BRANCH}:refs/remotes/origin/${BRANCH}"
   REMOTE_OID="$(git rev-parse "refs/remotes/origin/${BRANCH}")"
   printf 'branch=%s remote_oid=%s\n' "$BRANCH" "$REMOTE_OID"
   ```
   Keep the printed OID unchanged through the rebase. Do not recompute it after rewriting history; if commands run in another shell, substitute the printed OID literally in the push command below.
2. Identify the original commit that introduced the code being fixed (use `git log --oneline`, `git blame`, or the PR commit list).
3. Make the code edits in your working tree.
4. Stage only the relevant files (`git add <specific-paths>`), then create a fixup commit:
   ```bash
   git add <specific-paths>
   git commit --fixup=<original-commit-sha>
   ```
   This produces a commit whose message is `fixup! <original-subject>` — no body, no signoff line needed at this stage.
5. Squash the fixup commit into its target during the next push. Use `GIT_SEQUENCE_EDITOR=true` so the autosquash todo list is auto-accepted without opening an interactive editor:
   ```bash
   GIT_SEQUENCE_EDITOR=true git rebase -i --autosquash <base-branch>
   ```
   With `--autosquash`, Git automatically reorders fixup commits next to their targets and marks them `squash`/`fixup` in the todo list. `GIT_SEQUENCE_EDITOR=true` makes the editor "save-and-quit" non-interactively, so this command runs to completion without human input.
   - **Important**: This is the **only** permitted form of `git rebase -i` in this workflow. Free-form interactive rebase (without `--autosquash` or without `GIT_SEQUENCE_EDITOR=true`) is NOT supported.
6. Force-push with a lease bound to the saved OID, not the mutable remote-tracking ref:
   ```bash
   git push origin "HEAD:refs/heads/${BRANCH}" \
     "--force-with-lease=refs/heads/${BRANCH}:${REMOTE_OID}"
   ```
   If the lease fails, the remote branch changed after step 1. Stop, fetch and review the new commits, then reconcile them before attempting another rewrite. Never refresh the lease only to make the push pass.
7. The original commit's `-s` sign-off is preserved through the squash. Do not re-sign manually.

**If the fixup target is ambiguous** (e.g., touches code added across multiple commits), prefer folding the fix into the most recent related commit and rebasing, rather than introducing a new "fix review" commit.

**Exception — genuinely new commits during review:** If review feedback reveals a genuinely new, self-contained change that does not belong to any existing commit (e.g., reviewer asks for a new test file or a new helper module that was not part of the original change), it may be added as a new commit — but the total PR commit count must still stay ≤ 5. If already at 5, fold the new change into the most related existing commit via `--fixup`.

### 4. Verification Gate for Dependency / Setup Changes

- If the staged changes modify a ROS package `package.xml` dependency declaration (`depend`, `exec_depend`, `build_depend`, `test_depend`, etc.), or global setup/build workflow files such as `scripts/setup.sh`, `scripts/build.sh`, `scripts/setup/platforms/*.sh`, `scripts/setup/verify_env.sh`, `scripts/install_ros.sh`, top-level `CMakeLists.txt`, top-level `pyproject.toml`, or `requirements/*.txt` (pip dependency files that affect setup/install), the eventual PR description must include real Ubuntu 22.04 and openEuler Embedded Docker `setup.sh + build.sh` verification results.
- ROS package-local `setup.py` changes do **not** by themselves trigger this dual-platform setup/build gate. Examples that do not trigger it alone: console entry points, Python package metadata, or Python-only `install_requires` edits.
- When the gate is triggered, ask the user before Docker execution whether the PR is temporary WIP or ready for reviewer inspection. The question MUST be asked by invoking the interactive ask-user tool (the `question` tool in opencode), not by casually mentioning it in prose and not by assuming a default answer. Pass the explicit answer as `--pr-stage wip|review` to the PR workflow. WIP normalizes the title to `[WIP] <title>` and defers both Docker runs; it does not waive DCO, AI disclosure, other tests, or CI. Review stage removes `[WIP]`, records `git rev-parse HEAD^{tree}`, and makes both Docker skills test an isolated snapshot. The PR description then contains a structured `## Docker Verification` block with mode, Verified inputs, Tested source tree, and Docker environment fields. When only non-gated source changes, the workflow reuses prior evidence as `reused-environment` mode.

## Execution Steps

### Status Determination
If user explicitly requests **local commit only** (e.g., "commit to local", "only commit no push"), execute **only Phase 1, 2, and local commit part of Phase 3**. Skip push to remote, PR link, and PR description generation.

### Phase 1: Check and Summarize
1. Run `git status` in root directory and `libs/lerobot` separately.
2. Summarize pending changes to user.

### Phase 2: Compose Commit Message
1. Help user draft commit message (Title, Body, Footer) following specifications above.
2. **Validate**: Check title length, format, blank lines, Chinese characters, DCO sign-off, and AI metadata when AI
   participated. Confirm the real model name and version; do not infer or abbreviate it. Verify sign-off
   provenance: no hand-written `Signed-off-by` lines; the `-s` sign-off must match `git config user.name <user.email>`.
3. **Confirm staging scope**: Show the user which files will be committed, explicitly noting any excluded files (especially `libs/lerobot`).
4. **Check verification gate**: Inspect the staged file list for ROS package `package.xml` dependency changes or global setup/build workflow changes. If triggered, ask whether this is a `[WIP]` PR or ready for reviewer inspection before invoking Docker — via the interactive ask-user tool (`question` in opencode), never a prose question or an assumed default.

### Phase 3: Execute Commit and Push

**Important**: `libs/lerobot` is patch-managed and must NOT be committed to the root repo during normal commits. Only commit the files related to the current task.

For root repository:
1. Stage only the relevant files: `git add <specific-paths>` (NOT `git add .`).
2. If `libs/lerobot` was accidentally staged, unstage it: `git restore --staged libs/lerobot`.
3. Verify staging area with `git diff --cached --stat` before committing.
4. **Lint staged Python files only**: Run ruff check and format **only on the files you are about to commit**, NOT on the entire project:
   ```bash
   # Get the list of staged Python files
   STAGED_PY=$(git diff --cached --name-only --diff-filter=ACM -- '*.py')
   if [ -n "$STAGED_PY" ]; then
     ruff check $STAGED_PY && ruff format $STAGED_PY
   fi
   ```
   If ruff auto-fixed or reformatted any file, re-stage it: `git add <fixed-files>`.
   **Never** run bare `ruff check --fix .` or `ruff format .` — that modifies unrelated files and pollutes the commit.
5. For an AI-assisted contribution, run `git commit -s` with
   `Co-Authored-By: <AI model name and version>` in the message body/footer. For a contribution with no AI-generated
   or AI-processed content, run `git commit -s` without that trailer.
6. If NOT "local commit only":
   - Execute `git push origin <branch>` for a normal fast-forward push. For amend/rebase pushes, capture the remote OID **before** rewriting and use the explicit OID-bound lease from §3; a bare `--force-with-lease` is not allowed.
   - **Get remote info**: Extract username and repo name via `git remote get-url origin`.
   - **Check for existing PR**: After push, check whether this branch already has an open PR on the target repo. If the push output or remote hook response contains a PR/MR URL, extract the PR number. Otherwise, query via `pr_management.py --fetch-info` or check AtomGit UI.
   - **If an existing PR is found**: The PR description is now stale. You **must** synchronize it:
     1. Run `python3 pr_management.py --pr <NUM> --fetch-info` to get full PR context (all commits + diff).
     2. Analyze all commits in the PR and regenerate a complete PR description (Chinese by default) covering all changes.
     3. Before updating, collect the `Co-Authored-By` models from **every** PR commit (including commits pushed by
        other sessions or agents) and pass the full union via `--ai-model`; the update script validates coverage
        against the remote PR commits and rejects missing models.
     4. If the PR context triggers the verification gate, ask for WIP/review stage. For WIP, keep `[WIP]` and skip Docker. For review, run both Docker skills against the latest remote head tree and include the structured `## Docker Verification` block. When inputs unchanged, the workflow auto-inserts a `reused-environment` block.
     4. Write the updated `description.json` and run `python3 pr_management.py --pr <NUM> --update-pr description.json --pr-stage <wip|review>`.
   - **If no existing PR**: Generate AtomGit PR link: `https://atomgit.com/<username>/IB_Robot/merge_requests/new?source_branch=<current-branch>` and compose PR description from commit message body.

For submodule `libs/lerobot` (only when user explicitly asks):
1. Change to directory -> `git add` -> `git commit -s`.
2. If NOT "local commit only":
   - Execute `git push origin <branch>`.
   - Record commit hash.
3. **Remind the user**: Changes in `libs/lerobot` should normally be exported as patches via `ibrobot-lerobot-patch` skill instead of being committed directly to the root repo.

## Common Commands Reference
- Push to personal fork: `git push origin <branch>`
- Force push (amend/rebase): `git push origin HEAD:refs/heads/<branch> --force-with-lease=refs/heads/<branch>:<saved-remote-oid>`; save the OID before rewriting as specified in §3
- Signed commit: `git commit -s` (opens editor or use -m flag)
- AI-assisted signed commit: include `Co-Authored-By: <AI model name and version>` before the sign-off generated by
  `git commit -s`
- Undo last commit (keep changes): `git reset --soft HEAD~1`
- Create fixup commit (review fix): `git commit --fixup=<original-sha>` (no `-s` needed)
- Autosquash fixup commits non-interactively: `GIT_SEQUENCE_EDITOR=true git rebase -i --autosquash <base-branch>`
- List commits in current branch vs base: `git log --oneline <base-branch>..HEAD`
- Count commits ahead of base: `git rev-list --count <base-branch>..HEAD`

## Review Fix Workflow (Quick Reference)

When the user asks to "修复 review 意见" / "按 PR 意见改一下" / "address reviewer feedback" and the fix is to code in an existing PR commit:

1. Fetch `origin/<branch>` and save its exact OID using the commands in §3. Keep that OID unchanged.
2. `git log --oneline <base-branch>..HEAD` — list current PR commits, pick the target SHA.
3. Edit the code, stage relevant files with `git add <specific-paths>`.
4. `git commit --fixup=<target-sha>` — creates a `fixup! ...` commit.
5. Check current PR commit count: `git rev-list --count <base-branch>..HEAD`.
6. `GIT_SEQUENCE_EDITOR=true git rebase -i --autosquash <base-branch>` — folds the fixup into its target.
7. `git push origin HEAD:refs/heads/<branch> --force-with-lease=refs/heads/<branch>:<saved-remote-oid>` — push only if the remote still has the saved OID.
8. Verify commit count stays ≤ 5 after autosquash: `git rev-list --count <base-branch>..HEAD`.

**Do NOT** do any of the following as part of a review fix:
- `git commit -s -m "fix: address review comments"` (creates a new noisy commit).
- Amending the target commit manually with `git commit --amend` (loses the clear fixup→target association; harder to review in `git log` before push).
- Adding a brand-new commit when the change actually belongs to an existing one.

If the user's review fix is genuinely a new self-contained change (see "Exception" in §3), fall back to the normal commit flow (§1–§2 of Execution Steps) and re-verify the total commit count ≤ 5.

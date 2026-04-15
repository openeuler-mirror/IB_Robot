#!/usr/bin/env python3
"""
AtomGit PR Review Workflow
支持三种模式：
1. --extract-info: 提取 PR 信息（输出JSON）- AI Agent 使用
2. --submit-review: 提交审查结果（从JSON读取）- AI Agent 使用
3. --auto: 自动审查（调用LLM）- CI 使用，需要配置 LLM
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from atomgit_sdk import AtomGitClient, CodeIssue, resolve_atomgit_context
from atomgit_sdk.utils import calculate_diff_position, add_line_numbers

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from comment_formatter import CommentFormatter
from llm_reviewer import LLMCodeReviewer


class CodeReviewer:
    """代码审查器"""

    def __init__(self, client: AtomGitClient, formatter: CommentFormatter):
        self.client = client
        self.formatter = formatter

    def extract_pr_info(self, pr_number: int, include_comments: bool = True) -> dict:
        """提取适合 review 场景的完整 PR 上下文"""
        pr = self.client.get_pull_request(pr_number)
        files = self.client.get_pr_files(pr_number)
        commits = self.client.get_pr_commits(pr_number)
        comments = [] if not include_comments else self.client.get_pr_comments(pr_number)
        head_sha = pr.get("head", {}).get("sha", "HEAD")
        additions = sum(f.get("additions", 0) for f in files)
        deletions = sum(f.get("deletions", 0) for f in files)

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

        inline_comment_count = sum(
            1 for comment in comments if comment.get("path") or comment.get("diff_file")
        )
        unresolved_comment_count = sum(
            1 for comment in comments if not comment.get("resolved_at")
        )

        return {
            "fetch_time": datetime.now().isoformat(),
            "pr": {
                "number": pr.get("number"),
                "title": pr.get("title"),
                "author": pr.get("user", {}).get("login"),
                "state": pr.get("state"),
                "branch": f"{pr.get('head', {}).get('ref')} → {pr.get('base', {}).get('ref')}",
                "head_sha": head_sha,
                "stats": {
                    "files_changed": len(changed_files),
                    "commits": len(commits),
                    "comments": len(comments),
                    "inline_comments": inline_comment_count,
                    "unresolved_comments": unresolved_comment_count,
                    "additions": additions,
                    "deletions": deletions,
                },
                "changed_files": changed_files,
            },
            "commits": [
                {
                    "sha": commit.get("sha", ""),
                    "author": commit.get("commit", {}).get("author", {}).get("name", ""),
                    "message": commit.get("commit", {}).get("message", ""),
                }
                for commit in commits
            ],
            "comments": comments,
        }

    def load_issues_from_json(self, json_path: str) -> List[CodeIssue]:
        """从 JSON 文件加载问题"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        issues = []
        for item in data:
            issue = CodeIssue(
                file=item.get("file", ""),
                line=item.get("line", 0),
                type=item.get("type", "bug"),
                severity=item.get("severity", "warning"),
                confidence=item.get("confidence", 80),
                title=item.get("title", ""),
                description=item.get("description", ""),
                context_code=item.get("contextCode") or item.get("context_code"),
                fix_code=item.get("fix", {}).get("code")
                if isinstance(item.get("fix"), dict)
                else item.get("fix_code"),
                fix_explanation=item.get("fix", {}).get("explanation")
                if isinstance(item.get("fix"), dict)
                else item.get("fix_explanation"),
            )
            issues.append(issue)

        return issues

    def submit_issues(self, pr_number: int, issues: List[CodeIssue]) -> Dict:
        """提交问题到 PR"""
        pr = self.client.get_pull_request(pr_number)
        diffs = self.client.get_pr_diff(pr_number)

        issues = self.formatter.deduplicate(issues)

        positions = {}
        for issue in issues:
            if issue.file not in positions:
                positions[issue.file] = {}

            diff_info = diffs.get(issue.file, {})
            is_new_file = diff_info.get("status") == "added"
            patch = diff_info.get("patch", "")
            position = calculate_diff_position(patch, issue.line, is_new_file)
            if position is not None:
                positions[issue.file][issue.line] = position

        comments = self.formatter.format_issues(issues, positions)

        summary = self.formatter.format_summary(issues, pr_number, pr.get("title", ""))
        self.client.submit_pr_comment(pr_number, summary)
        print(f"✅ 已提交摘要评论\n")

        if comments:
            results = self.client.submit_batch_comments(pr_number, comments)
            success_count = sum(1 for r in results if r["success"])

            print(f"✅ 提交 {success_count}/{len(results)} 条评论\n")

            for result in results:
                if result["success"]:
                    print(f"  ✅ {result['comment']['path']} → {result['comment_url']}")
                else:
                    print(f"  ❌ {result['comment']['path']} - {result['error']}")
        else:
            print("⚠️  没有符合条件的问题需要提交\n")

        return {
            "total_issues": len(issues),
            "submitted_comments": len(comments),
            "summary_submitted": True,
        }


def mode_extract_info(args, reviewer: CodeReviewer):
    """模式1: 提取 PR 信息（AI Agent 使用）"""
    print("\n" + "=" * 60)
    print("📥 模式: 提取 PR 信息")
    print("=" * 60)

    pr_info = reviewer.extract_pr_info(args.pr, include_comments=not args.no_comments)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_name = reviewer.client.config.repo.lower().replace("-", "_")
    output_file = output_dir / f"{repo_name}_pr_{args.pr}_info.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pr_info, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 已保存到: {output_file}")
    print(f"\n📊 变更摘要:")
    print(f"   标题: {pr_info['pr']['title']}")
    print(f"   作者: {pr_info['pr']['author']}")
    print(f"   分支: {pr_info['pr']['branch']}")
    print(f"   文件: {pr_info['pr']['stats']['files_changed']} 个")
    print(f"   提交: {pr_info['pr']['stats']['commits']} 个")
    print(f"   评论: {pr_info['pr']['stats']['comments']} 条")
    if not args.no_comments:
        print(f"   未解决评论: {pr_info['pr']['stats']['unresolved_comments']} 条")

    print("\n💡 下一步:")
    print("  AI Agent 应该:")
    print("  1. 读取此文件并进行代码审查")
    print("  2. 结合 changed_files、commits 和 comments 生成 issues.json")
    print("  3. ⚠️ 将审查结果以用户可读的格式展示给用户确认")
    print("  4. 用户确认后，运行提交命令")
    print(
        f"\n     python3 pr_review.py --pr {args.pr} --submit-review issues.json --ai-model <your-model-name>"
    )


def mode_submit_review(args, reviewer: CodeReviewer):
    """模式2: 提交审查结果（AI Agent 使用）"""
    print("\n" + "=" * 60)
    print("📤 模式: 提交审查结果")
    print("=" * 60)

    print(f"\n📂 从 JSON 加载问题: {args.submit_review}\n")

    issues = reviewer.load_issues_from_json(args.submit_review)
    print(f"📝 加载了 {len(issues)} 个问题\n")

    if args.dry_run:
        print("ℹ️  Dry run 模式：将显示提交计划但不执行\n")
        for issue in issues:
            if issue.confidence >= args.threshold:
                print(f"  - {issue.file}:{issue.line} [{issue.severity}] {issue.title}")
        print("")
        return

    result = reviewer.submit_issues(args.pr, issues)

    print(f"\n" + "=" * 60)
    print(f"✅ 审查完成")
    print("=" * 60 + "\n")
    print(f"📊 统计:")
    print(f"   总问题数: {result['total_issues']}")
    print(f"   提交评论数: {result['submitted_comments']}")
    print(f"\n🔗 PR 链接: {reviewer.client.get_pr_url(args.pr)}\n")


def mode_auto(args, client: AtomGitClient, reviewer: CodeReviewer, config: dict):
    """模式3: 自动审查（CI 使用，需要 LLM 配置）"""
    print("\n" + "=" * 60)
    print("🤖 模式: 自动审查（LLM驱动）")
    print("=" * 60)

    if not config.get("anthropic", {}).get("apiKey"):
        print("\n❌ 自动模式需要配置 Anthropic API Key")
        print("   请在 config.json 中添加:")
        print("   {")
        print('     "anthropic": {')
        print('       "apiKey": "sk-ant-..."')
        print("     }")
        print("   }")
        print("\n或者使用手动模式（AI Agent 调用）:")
        print(f"   python3 pr_review.py --pr {args.pr} --extract-info")
        return

    pr_info = client.get_pull_request(args.pr)
    head_sha = pr_info.get("head", {}).get("sha", "HEAD")

    llm_reviewer = LLMCodeReviewer(
        api_key=config["anthropic"]["apiKey"],
        base_url=config["anthropic"].get("baseUrl", ""),
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
    )

    print(f"\n📝 获取 PR 文件变更...")
    files = client.get_pr_files(args.pr)

    all_issues = []

    for i, file_info in enumerate(files, 1):
        file_path = file_info["filename"]

        if not file_path.endswith(".py"):
            continue

        if "test" in file_path.lower():
            continue

        print(f"\n[{i}/{len(files)}] 审查 {file_path}")

        try:
            content = client.get_file_content(file_path, head_sha)
            diff = file_info.get("patch", "")

            print("  ⏳ 调用 LLM 进行审查...")
            issues = llm_reviewer.review_file(file_path, content, diff)

            if issues:
                print(f"  ✓ 发现 {len(issues)} 个问题")
                all_issues.extend(issues)
            else:
                print(f"  ✓ 未发现问题")

        except Exception as e:
            print(f"  ✗ 审查失败: {e}")

    if args.dry_run:
        print(f"\n" + "=" * 60)
        print("⚠️  Dry run 模式，未提交评论")
        print("=" * 60)
        print(f"\n发现 {len(all_issues)} 个问题：")
        for issue in all_issues:
            print(f"  - {issue.file}:{issue.line} [{issue.severity}] {issue.title}")
        return

    if all_issues:
        print(f"\n📦 提交 {len(all_issues)} 个审查结果...")
        result = reviewer.submit_issues(args.pr, all_issues)

        print(f"\n" + "=" * 60)
        print("✅ 审查完成")
        print("=" * 60)
        print(f"\n📊 统计:")
        print(
            f"   审查文件: {len([f for f in files if f['filename'].endswith('.py')])} 个"
        )
        print(f"   发现问题: {result['total_issues']} 个")
        print(f"   提交评论: {result['submitted_comments']} 条")
    else:
        pr = client.get_pull_request(args.pr)
        summary = reviewer.formatter.format_summary([], args.pr, pr.get("title", ""))
        client.submit_pr_comment(args.pr, summary)

        print(f"\n" + "=" * 60)
        print("✅ 审查完成 - 未发现问题")
        print("=" * 60)

    print(f"\n🔗 PR 链接: {client.get_pr_url(args.pr)}\n")


def main():
    parser = argparse.ArgumentParser(
        description="AtomGit 代码审查",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--pr", type=int, help="PR 编号，可由 --url 自动解析")

    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--extract-info",
        action="store_true",
        help="模式1: 提取 PR 信息（AI Agent 使用）",
    )
    mode_group.add_argument(
        "--submit-review",
        type=str,
        metavar="JSON_FILE",
        help="模式2: 提交审查结果（AI Agent 使用）",
    )
    mode_group.add_argument(
        "--auto", action="store_true", help="模式3: 自动审查（CI 使用，需要 LLM 配置）"
    )

    parser.add_argument(
        "--config", type=str, default="config.json", help="配置文件路径"
    )
    parser.add_argument("--owner", type=str, help="目标仓库 owner，覆盖 config.json")
    parser.add_argument("--repo", type=str, help="目标仓库 repo，覆盖 config.json")
    parser.add_argument(
        "--url", type=str, help="PR 链接，用于自动解析 owner/repo/PR 编号"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./tmp", help="输出目录 (默认: ./tmp)"
    )
    parser.add_argument(
        "--no-comments",
        action="store_true",
        help="在 --extract-info 模式下跳过抓取现有 PR 评论",
    )
    parser.add_argument("--threshold", type=int, default=80, help="置信度阈值")
    parser.add_argument(
        "--ai-model",
        type=str,
        default="ai",
        help="AI模型名称，用于签名 (默认: ai)",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅显示计划，不提交")

    parser.add_argument(
        "--llm-provider",
        type=str,
        default="anthropic",
        help="LLM 提供商（仅 --auto 模式，默认: anthropic）",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="LLM 模型名称（仅 --auto 模式，默认: claude-sonnet-4-20250514）",
    )

    args = parser.parse_args()

    if args.ai_model == "ai":
        print("\n⚠️  警告: 未指定 --ai-model 参数，将使用默认值 'ai'")
        print("   建议指定真实模型名称，例如：")
        print("   --ai-model claude-sonnet-4")
        print("   --ai-model gpt-4")
        print("   --ai-model gemini-pro")
        print()

    print("\n" + "=" * 60)
    print("🔍 AtomGit 代码审查工具")
    print("=" * 60)

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"\n❌ 配置文件不存在: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 加载配置失败: {e}")
        sys.exit(1)

    try:
        sdk_config, parsed_url = resolve_atomgit_context(
            args.config, owner=args.owner, repo=args.repo, url=args.url
        )
    except Exception as e:
        print(f"\n❌ 解析仓库上下文失败: {e}")
        sys.exit(1)

    if args.pr is None:
        args.pr = parsed_url.get("pr_number")
    if args.pr is None:
        print("\n❌ 缺少 PR 编号。请通过 --pr 指定，或传入包含 PR 编号的 --url。")
        sys.exit(1)

    client = AtomGitClient(sdk_config)
    formatter = CommentFormatter(
        confidence_threshold=args.threshold, ai_model=args.ai_model
    )
    reviewer = CodeReviewer(client, formatter)

    print(f"\n📋 PR: #{args.pr}")
    print(f"🏠 仓库: {client.config.owner}/{client.config.repo}")
    if args.url:
        print(f"🔗 链接: {args.url}")
    print(f"🤖 AI模型: {args.ai_model}")

    if args.auto:
        print(f"🧠 LLM模型: {args.llm_model} (provider: {args.llm_provider})")
        print("📦 模式: 自动（CI 模式，Skill 内部调用 LLM）")
    elif args.extract_info:
        print("📥 模式: 提取信息（AI Agent 模式）")
    elif args.submit_review:
        print("📤 模式: 提交审查（AI Agent 模式）")
    else:
        args.extract_info = True
        print("📥 模式: 提取信息（默认，AI Agent 模式）")

    if args.dry_run:
        print("⚠️  Dry Run 模式（仅显示计划）")

    if args.extract_info:
        mode_extract_info(args, reviewer)
    elif args.submit_review:
        mode_submit_review(args, reviewer)
    elif args.auto:
        mode_auto(args, client, reviewer, config)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
AtomGit Architecture Review Workflow
支持三种模式：
1. --extract-info: 提取 PR 信息（输出JSON）- AI Agent 使用
2. --submit-review: 提交审查结果（从JSON读取）- AI Agent 使用
3. --auto: 自动审查（调用LLM）- CI 使用，需要配置 LLM
"""

import argparse
import json
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from atomgit_sdk import ArchitectureIssue, AtomGitClient, AtomGitConfig
from atomgit_sdk.utils import add_line_numbers
from comment_formatter import CommentFormatter
from llm_architecture_reviewer import LLMArchitectureReviewer


class ArchitectureReviewer:
    """架构审查器"""

    def __init__(self, client: AtomGitClient, formatter: CommentFormatter):
        self.client = client
        self.formatter = formatter

    def _candidate_readme_paths(self, file_path: str) -> list[str]:
        """Return nearest README candidates for a repository path."""
        path = PurePosixPath(file_path)
        candidates = []
        for parent in path.parents:
            parent_text = str(parent)
            if parent_text in (".", ""):
                continue
            candidate = f"{parent_text}/README.md"
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def get_readme_context(self, file_path: str, ref: str) -> dict:
        """Fetch nearest package README context for a changed file."""
        candidates = self._candidate_readme_paths(file_path)
        errors = []
        for candidate in candidates:
            try:
                content = self.client.get_file_content(candidate, ref)
                return {
                    "path": candidate,
                    "content": add_line_numbers(content),
                    "candidates": candidates,
                }
            except Exception as e:
                errors.append(f"{candidate}: {e}")

        return {
            "path": None,
            "content": "",
            "candidates": candidates,
            "errors": errors,
        }

    def extract_pr_info(self, pr_number: int) -> dict:
        """提取 PR 信息"""
        pr = self.client.get_pull_request(pr_number)
        files = self.client.get_pr_files(pr_number)
        head_sha = pr.get("head", {}).get("sha", "HEAD")

        changed_files = []
        for f in files:
            filename = f.get("filename") or ""
            is_arch_relevant = filename.endswith(".py") or filename.endswith("README.md")
            if f.get("status") != "removed" and is_arch_relevant:
                file_data = {
                    "filename": filename,
                    "status": f.get("status"),
                    "additions": f.get("additions", 0),
                    "deletions": f.get("deletions", 0),
                    "patch": f.get("patch"),
                }

                # 获取文件内容（使用 PR 的 head SHA）
                try:
                    content = self.client.get_file_content(filename, head_sha)
                    file_data["content"] = add_line_numbers(content)
                except Exception as e:
                    file_data["content"] = f"# Error fetching content: {e}"

                file_data["readme_context"] = self.get_readme_context(filename, head_sha)
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

    def load_issues_from_json(self, json_path: str) -> list[ArchitectureIssue]:
        """从 JSON 文件加载问题"""
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        issues = []
        for item in data:
            issue = ArchitectureIssue(
                file=item.get("file", ""),
                line=item.get("line", 0),
                title=item.get("title", ""),
                description=item.get("description", ""),
                severity=item.get("severity", "warning"),
                pillar=item.get("pillar", "python"),
                fix=item.get("fix"),
                context_code=item.get("context_code"),
            )
            issues.append(issue)

        return issues

    def submit_review(self, pr_number: int, issues: list[ArchitectureIssue]) -> None:
        """提交审查结果"""
        if not issues:
            summary = self.formatter.format_summary(issues)
            self.client.submit_pr_comment(pr_number, summary)
            print(f"✅ 提交架构审查通过评论到 PR #{pr_number}")
            return

        pr_files = self.client.get_pr_files(pr_number)
        changed_filenames = {f.get("filename") for f in pr_files}

        inline_issues = []
        pr_level_issues = []
        for issue in issues:
            if issue.file in changed_filenames:
                inline_issues.append(issue)
            else:
                pr_level_issues.append(issue)

        if inline_issues:
            comments = self.formatter.format_issues(inline_issues)
            results = self.client.submit_batch_comments(pr_number, comments)

            success_count = sum(1 for r in results if r["success"])
            print(f"✅ 提交 {success_count}/{len(results)} 条行内评论到 PR #{pr_number}")

            for result in results:
                if not result["success"]:
                    print(f"  ❌ 失败: {result['comment']['path']} - {result['error']}")

        if pr_level_issues:
            body = self.formatter.format_pr_level_issues(pr_level_issues)
            try:
                self.client.submit_pr_comment(pr_number, body)
                print(f"✅ 提交 {len(pr_level_issues)} 条 PR 评论到 PR #{pr_number}（非变更文件，降级为 PR 级评论）")
            except Exception as e:
                print(f"  ❌ PR 评论提交失败: {e}")


def mode_extract_info(args, reviewer: ArchitectureReviewer):
    """模式1: 提取 PR 信息（AI Agent 使用）"""
    print("\n" + "=" * 60)
    print("📥 模式: 提取 PR 信息")
    print("=" * 60)

    pr_info = reviewer.extract_pr_info(args.pr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 从配置中获取仓库名称
    repo_name = reviewer.client.config.repo.lower().replace("-", "_")
    output_file = output_dir / f"{repo_name}_pr_{args.pr}_arch_info.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pr_info, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 已保存到: {output_file}")
    print("\n📊 变更摘要:")
    print(f"   标题: {pr_info['pr']['title']}")
    print(f"   作者: {pr_info['pr']['author']}")
    print(f"   分支: {pr_info['pr']['branch']}")
    print(f"   架构相关文件: {len(pr_info['pr']['changed_files'])} 个")

    print("\n💡 下一步:")
    print("  AI Agent 应该:")
    print("  1. 读取此文件并进行架构审查")
    print("  2. 检查是否符合 IB_Robot 架构四大支柱")
    print("  3. 对比每个 changed_files[].readme_context，检查 README 是否与代码同步")
    print("  4. 生成 arch-issues.json（包含所有架构问题和建议）")
    print("  5. ⚠️ 将审查结果以用户可读的格式展示给用户确认")
    print("  6. 用户确认后，运行提交命令")
    print(
        f"\n     python3 architecture_review.py --pr {args.pr} --submit-review arch-issues.json --ai-model <your-model-name>"
    )


def mode_submit_review(args, reviewer: ArchitectureReviewer):
    """模式2: 提交审查结果（AI Agent 使用）"""
    print("\n" + "=" * 60)
    print("📤 模式: 提交审查结果")
    print("=" * 60)

    print(f"\n📂 从 JSON 加载问题: {args.submit_review}\n")

    issues = reviewer.load_issues_from_json(args.submit_review)
    print(f"📝 加载了 {len(issues)} 个架构问题\n")

    if args.dry_run:
        print("ℹ️  Dry run 模式：将显示提交计划但不执行\n")
        for issue in issues:
            print(f"  - {issue.file}:{issue.line} [{issue.severity}] {issue.title}")
        print("")
        return

    reviewer.submit_review(args.pr, issues)

    print("\n" + "=" * 60)
    print("✅ 审查完成")
    print("=" * 60 + "\n")
    print("📊 统计:")
    print(f"   总问题数: {len(issues)}")
    print(f"\n🔗 PR 链接: {reviewer.client.get_pr_url(args.pr)}\n")


def mode_auto(args, client: AtomGitClient, reviewer: ArchitectureReviewer, config: dict):
    """模式3: 自动审查（CI 使用，需要 LLM 配置）"""
    print("\n" + "=" * 60)
    print("🤖 模式: 自动审查（LLM驱动）")
    print("=" * 60)

    # 检查 LLM 配置
    if not config.get("anthropic", {}).get("apiKey"):
        print("\n❌ 自动模式需要配置 Anthropic API Key")
        print("   请在 config.json 中添加:")
        print("   {")
        print('     "anthropic": {')
        print('       "apiKey": "sk-ant-..."')
        print("     }")
        print("   }")
        print("\n或者使用手动模式（AI Agent 调用）:")
        print(f"   python3 architecture_review.py --pr {args.pr} --extract-info")
        return

    # 创建 LLM 审查器
    llm_reviewer = LLMArchitectureReviewer(
        api_key=config["anthropic"]["apiKey"],
        base_url=config["anthropic"].get("baseUrl", ""),
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
    )

    # 获取 PR 信息以获取 head_sha
    pr_info = client.get_pull_request(args.pr)
    head_sha = pr_info.get("head", {}).get("sha", "HEAD")

    print("\n📝 获取 PR 文件变更...")
    files = client.get_pr_files(args.pr)

    all_issues = []

    for i, file_info in enumerate(files, 1):
        file_path = file_info["filename"]

        is_readme = file_path.endswith("README.md")
        is_python = file_path.endswith(".py")

        # 只审查 Python 文件和 README
        if not (is_python or is_readme):
            continue

        # 跳过测试文件
        if is_python and "test" in file_path.lower():
            continue

        # 跳过 __init__.py
        if is_python and file_path.endswith("__init__.py"):
            continue

        print(f"\n[{i}/{len(files)}] 审查 {file_path}")

        try:
            # 获取文件内容（使用 head_sha）
            content = client.get_file_content(file_path, head_sha)
            diff = file_info.get("patch", "")
            readme_context = reviewer.get_readme_context(file_path, head_sha)

            # 调用 LLM 审查
            print("  ⏳ 调用 LLM 进行架构审查...")
            issues = llm_reviewer.review_file(
                file_path,
                content,
                diff,
                readme_context=readme_context,
            )

            if issues:
                print(f"  ✓ 发现 {len(issues)} 个架构问题")
                all_issues.extend(issues)
            else:
                print("  ✓ 未发现架构问题")

        except Exception as e:
            print(f"  ✗ 审查失败: {e}")

    if args.dry_run:
        print("\n" + "=" * 60)
        print("⚠️  Dry run 模式，未提交评论")
        print("=" * 60)
        print(f"\n发现 {len(all_issues)} 个架构问题：")
        for issue in all_issues:
            print(f"  - {issue.file}:{issue.line} [{issue.severity}] {issue.title}")
        return

    # 提交审查结果
    if all_issues:
        print(f"\n📦 提交 {len(all_issues)} 个架构审查结果...")
        reviewer.submit_review(args.pr, all_issues)

        print("\n" + "=" * 60)
        print("✅ 审查完成")
        print("=" * 60)
        print("\n📊 统计:")
        print(
            f"   审查文件: {len([f for f in files if f['filename'].endswith('.py') and 'test' not in f['filename'].lower()])} 个"
        )
        print(f"   发现问题: {len(all_issues)} 个")
    else:
        # 提交通过评论
        print("\n📦 提交架构审查通过评论...")
        reviewer.submit_review(args.pr, [])

        print("\n" + "=" * 60)
        print("✅ 审查完成 - 未发现架构问题")
        print("=" * 60)

    print(f"\n🔗 PR 链接: {client.get_pr_url(args.pr)}\n")


def main():
    parser = argparse.ArgumentParser(
        description="AtomGit 架构审查",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 必需参数
    parser.add_argument("--pr", type=int, required=True, help="PR 编号")

    # 模式选择（互斥）
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
    mode_group.add_argument("--auto", action="store_true", help="模式3: 自动审查（CI 使用，需要 LLM 配置）")

    # 通用参数
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--output-dir", type=str, default="./tmp", help="输出目录 (默认: ./tmp)")
    parser.add_argument(
        "--ai-model",
        type=str,
        default="ai",
        help="AI模型名称，用于签名 (默认: ai)",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅显示计划，不提交")

    # 自动模式专用参数（LLM 配置）
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

    # 警告：如果使用默认的 ai-model，提醒指定真实模型名称
    if args.ai_model == "ai":
        print("\n⚠️  警告: 未指定 --ai-model 参数，将使用默认值 'ai'")
        print("   建议指定真实模型名称，例如：")
        print("   --ai-model claude-sonnet-4")
        print("   --ai-model gpt-4")
        print("   --ai-model gemini-pro")
        print()

    print("\n" + "=" * 60)
    print("🔍 AtomGit 架构审查工具")
    print("=" * 60)

    # 加载配置
    try:
        with open(args.config, encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"\n❌ 配置文件不存在: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 加载配置失败: {e}")
        sys.exit(1)

    # 创建 API 实例
    api = AtomGitClient(AtomGitConfig.from_json(args.config))
    formatter = CommentFormatter(ai_model=args.ai_model)
    reviewer = ArchitectureReviewer(api, formatter)

    print(f"\n📋 PR: #{args.pr}")
    print(f"🏠 仓库: {api.config.owner}/{api.config.repo}")
    print(f"🤖 AI模型: {args.ai_model}")

    # 检查模式
    if args.auto:
        print(f"🧠 LLM模型: {args.llm_model} (provider: {args.llm_provider})")
        print("📦 模式: 自动（CI 模式，Skill 内部调用 LLM）")
    elif args.extract_info:
        print("📥 模式: 提取信息（AI Agent 模式）")
    elif args.submit_review:
        print("📤 模式: 提交审查（AI Agent 模式）")
    else:
        # 默认：提取信息模式（AI Agent 友好）
        args.extract_info = True
        print("📥 模式: 提取信息（默认，AI Agent 模式）")

    if args.dry_run:
        print("⚠️  Dry Run 模式（仅显示计划）")

    # 根据模式执行
    if args.extract_info:
        mode_extract_info(args, reviewer)
    elif args.submit_review:
        mode_submit_review(args, reviewer)
    elif args.auto:
        mode_auto(args, api, reviewer, config)


if __name__ == "__main__":
    main()

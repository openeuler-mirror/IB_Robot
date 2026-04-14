#!/usr/bin/env python3
"""
AtomGit SDK 冒烟测试
重点验证 diff 位置计算、URL 解析与核心对象初始化。
"""

import sys


def print_section(title: str) -> None:
    print(f"\n[{title}]")
    print("-" * 80)


def main() -> int:
    print("=" * 80)
    print("AtomGit SDK Smoke Test")
    print("=" * 80)

    print_section("TEST 1: SDK Import Check")
    try:
        from atomgit_sdk import (
            ArchitectureIssue,
            AtomGitClient,
            AtomGitConfig,
            CodeIssue,
            FixResult,
            PRService,
            parse_atomgit_url,
        )
        from atomgit_sdk.utils import calculate_diff_position
    except ImportError as exc:
        print(f"✗ Import failed: {exc}")
        return 1
    print("✓ All SDK modules imported successfully")

    passed = 0
    failed = 0

    print_section("TEST 2: Diff Position Calculation")
    diff_cases = [
        {
            "name": "Simple modification",
            "patch": """@@ -10,5 +10,6 @@
 context line 1
 context line 2
-old line
+new line
 context line 3""",
            "line_number": 12,
            "is_new_file": False,
            "expect_none": False,
        },
        {
            "name": "New file",
            "patch": "",
            "line_number": 5,
            "is_new_file": True,
            "expected": 5,
        },
        {
            "name": "Empty patch (not new file)",
            "patch": "",
            "line_number": 10,
            "is_new_file": False,
            "expected": None,
        },
    ]
    for case in diff_cases:
        result = calculate_diff_position(
            case["patch"], case["line_number"], case["is_new_file"]
        )
        if case.get("expect_none") is False:
            success = result is not None
        else:
            success = result == case["expected"]
        if success:
            print(f"✓ {case['name']}: {result}")
            passed += 1
        else:
            print(f"✗ {case['name']}: got {result}")
            failed += 1

    print_section("TEST 3: URL Parsing")
    url_cases = [
        (
            "https://atomgit.com/example-org/demo-repo/pulls/123",
            {"owner": "example-org", "repo": "demo-repo", "pr_number": 123},
        ),
        (
            "https://atomgit.com/example-org/demo-repo",
            {"owner": "example-org", "repo": "demo-repo"},
        ),
        (
            "https://atomgit.com/example-org/demo-repo/tree/feat/new-feature",
            {
                "owner": "example-org",
                "repo": "demo-repo",
                "branch": "feat/new-feature",
            },
        ),
        (
            "https://gitcode.com/example-org/demo-repo/merge_requests/76",
            {"owner": "example-org", "repo": "demo-repo", "pr_number": 76},
        ),
        (
            "https://atomgit.com/example-org/demo-repo/issues/12",
            {"owner": "example-org", "repo": "demo-repo", "issue_number": 12},
        ),
    ]
    for url, expected in url_cases:
        result = parse_atomgit_url(url)
        if result == expected:
            print(f"✓ {url}")
            passed += 1
        else:
            print(f"✗ {url}: expected {expected}, got {result}")
            failed += 1

    print_section("TEST 4: Data Models and Services")
    try:
        code_issue = CodeIssue(
            file="test.py",
            line=10,
            title="Test Issue",
            description="Test description",
            type="bug",
            severity="error",
            confidence=90,
        )
        arch_issue = ArchitectureIssue(
            file="config.yaml",
            line=5,
            title="Architecture Violation",
            description="SSOT violation",
            pillar="ssot",
            severity="warning",
        )
        fix_result = FixResult(
            has_fix=True,
            file_path="test.py",
            fix_description="Remove duplicate code",
            original_code="duplicate",
            fixed_code="unique",
        )
        config = AtomGitConfig(
            token="test-token",
            owner="test-owner",
            repo="test-repo",
            base_url="https://api.atomgit.com",
        )
        client = AtomGitClient(config)
        PRService(client)
        print(f"✓ CodeIssue created: {code_issue.title}")
        print(f"✓ ArchitectureIssue created: {arch_issue.title}")
        print(f"✓ FixResult created: {fix_result.fix_description}")
        print(f"✓ AtomGitConfig created: {config.owner}/{config.repo}")
        print("✓ AtomGitClient initialized")
        print("✓ PRService initialized")
        passed += 6
    except Exception as exc:
        print(f"✗ Model or service initialization failed: {exc}")
        failed += 1

    print("\n" + "=" * 80)
    print("SMOKE TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {passed + failed}")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")

    if failed:
        print(f"\n⚠️  {failed} test(s) failed. Please fix before migration.")
        return 1

    print("\n🎉 All smoke tests passed! SDK is ready for migration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

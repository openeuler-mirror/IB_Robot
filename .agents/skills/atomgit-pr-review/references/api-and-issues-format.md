# API 参考与 issues.json 格式

## When to Read

- 需要提取 PR 上下文或分阶段读取时
- 完成三步法后需要构造 issues.json 时
- 人类确认后需要提交审查结果时
- 处理大文件 JSON 时

## ⚠️ 依赖准备

本 skill 依赖 PyPI 包 `atomgit-sdk`，其 Python 导入模块名为 `atomgit_sdk`。
仓库默认通过 `requirements/*.txt` 安装；如果当前环境未安装，请先运行
`./scripts/setup.sh`，或在当前 Python 环境中安装 `atomgit-sdk`。

## ⚠️ 文件读取说明

**输出文件位于项目 `./tmp` 目录，包含实现与评论，前两步禁止整文件读取。**
即使文件很小，也必须按主文档三步法分阶段投影：

```bash
# 第 1 步：仅问题背景；不包含 changed_files、comments 或自动检查细节
jq '{pr: (.pr | {number, title, body, head_sha}), commits: [.commits[] | {sha, message}]}' ./tmp/ib_robot_pr_123_info.json

# 第 2 步：展示独立方案后，第 3 步才允许读取实现和评论
jq '.pr.mandatory_review_checks, .pr.changed_files, .comments' ./tmp/ib_robot_pr_123_info.json
```

**PR 正文**（AI 声明、Verification 等）在 `.pr.body` 字段。

### 大文件处理技巧

以下命令仅在第 3 步使用；大文件使用 `jq` 提取特定文件信息：

```bash
# 列出所有变更文件
jq '.pr.changed_files[].filename' ./tmp/ib_robot_pr_123_info.json

# 提取特定文件的内容
jq '.pr.changed_files[] | select(.filename == "lib/api.py") | .content' ./tmp/ib_robot_pr_123_info.json

# 提取特定文件的 diff
jq '.pr.changed_files[] | select(.filename == "lib/api.py") | .patch.diff' ./tmp/ib_robot_pr_123_info.json

# 提取多个文件（支持通配符）
jq '.pr.changed_files[] | select(.filename | contains("lib/")) | {filename, content}' ./tmp/ib_robot_pr_123_info.json
```

## API 说明

### 提取 PR 信息

```bash
python3 pr_review.py --pr 123
```

**输出**: 项目临时目录 `./tmp/{repo}_pr_{number}_info.json`（例如：`./tmp/ib_robot_pr_123_info.json`）

**注意**: 
- 默认输出到项目 `./tmp` 目录，**不需要指定 `--output-dir`**
- 默认包含 `changed_files`、`commits` 和已有 `comments`
- 如果评论量太大，可追加 `--no-comments`

```json
{
  "pr": {
    "number": 123,
    "title": "...",
    "body": "...",
    "author": "...",
    "branch": "feature → main",
    "stats": {
      "files_changed": 3,
      "commits": 2,
      "comments": 5,
      "unresolved_comments": 2
    },
    "changed_files": [
      {
        "filename": "lib/api.py",
        "status": "modified",
        "patch": "...",
        "content": "..."
      }
    ]
  },
  "commits": [...],
  "comments": [...]
}
```

**⚠️ 重要**：提取的 JSON 文件已经包含了所有 diff（`patch`）、文件内容（`content`）以及已有 PR 评论。
- **不需要** `git fetch` 或 `git diff`
- **不需要** 切换分支或修改本地代码
- 完成并展示问题陈述与独立方案后，第 3 步再读取 `changed_files` 和 `comments`
- 第 3 步先读取 `.pr.mandatory_review_checks`。如果包含 `lerobot_gitlink_changed`，必须按
  `references/ibrobot-mandatory-checks.md` 验证其是否为完整、可获取、已迁移 patch stack 的上游基线升级
- 审查时要结合变更文件判断是否需要 README / 文档联动，以及是否触发双平台 Docker Verification 门禁；标准 `[WIP]` 标题暂缓 Docker 证据检查，移除前缀后恢复。正式检视时默认只检查 PR 描述里的开发者验证声明，除非用户明确要求 agent 实际执行验证，否则不得调用 docker verification skills
- 如果需要"回复某一条已有 review 意见"而不是提交新的审查结果，请切换到 `atomgit-review-resolution`，使用 `--reply-comment <comment_id>`；不要在本 skill 中伪造普通 PR 级评论。

### 提交审查结果

```bash
python3 pr_review.py --pr 123 --submit-review ./tmp/ib_robot_pr_123_issues.json --ai-model glm-5.2
```

**参数**：
- `--pr`: PR 编号
- `--owner`: 目标仓库 owner（可选，覆盖 `config.json`）
- `--repo`: 目标仓库 repo（可选，覆盖 `config.json`）
- `--url`: PR 链接（可选，自动解析 `owner/repo/pr_number`）
- `--no-comments`: 在提取信息模式下跳过抓取已有 PR 评论
- `--submit-review`: 审查结果 JSON 文件
- `--ai-model`: AI 模型名称（**必须指定真实模型名称**，用于签名）
- `--dry-run`: 仅显示计划

**⚠️ 重要**：`--ai-model` 参数**必须指定你的真实模型名称**，以便在评论中准确标识来源。

**常见模型名称**：
- `glm-5.2`
- `glm-5.1`
- `gpt-5.6-sol`
- `gpt-5.6-terra`
- `claude-fable-5`
- `claude-opus-5`

## issues.json 格式

**重要要求**：
1. **必须使用中文**输出所有内容
2. **必须包含修复方案**（fix_code 字段）
3. **文件保存到 ./tmp 目录**，文件名格式：`./tmp/ib_robot_pr_{number}_issues.json`

```json
[
  {
    "file": "lib/api.py",
    "line": 52,
    "type": "bug",
    "severity": "error",
    "confidence": 95,
    "title": "缺少异常处理",
    "description": "response.json() 可能抛出 JSONDecodeError",
    "context_code": "return response.json()",
    "fix_code": "try:\n    return response.json()\nexcept json.JSONDecodeError:\n    return {}",
    "fix_explanation": "添加异常处理避免程序崩溃"
  }
]
```

### 字段说明

| 字段 | 必填 | 说明 | 可选值 |
|------|------|------|--------|
| file | ✅ | 文件路径 | |
| line | ✅ | 行号 | |
| type | ✅ | 问题类型（中文） | `bug`, `security`, `performance`, `maintainability` |
| severity | ✅ | 严重程度（中文） | `error`, `warning`, `suggestion`, `info` |
| confidence | ✅ | 置信度 (0-100) | |
| title | ✅ | 问题标题（中文） | |
| description | ✅ | 详细描述（中文） | |
| context_code | ❌ | 相关代码 | |
| fix_code | ✅ | 修复代码（必须提供） | |
| fix_explanation | ✅ | 修复说明（中文） | |

## 配置

在项目根目录的 `config.json` 中：

```json
{
  "atomgit": {
    "token": "your_personal_access_token",
    "owner": "openEuler",
    "repo": "IB_Robot",
    "baseUrl": "https://api.atomgit.com"
  }
}
```

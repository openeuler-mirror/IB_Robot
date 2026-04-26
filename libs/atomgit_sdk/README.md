# AtomGit SDK

统一的 AtomGit API 封装 SDK，可用于 AtomGit / GitCode 上的 PR、Issue、Review 与评论修复流程。

## 特性

- **统一的 API 客户端**: 封装所有 AtomGit API 调用
- **API 文档目录**: 内置常用协作 endpoint catalog，并支持从官方文档同步完整 endpoint 列表
- **类型安全**: 使用 Pydantic 模型进行数据验证
- **Diff 解析**: 准确的 Diff 行号映射算法（已修复 bug）
- **高级服务**: PRService 和 RepairService 封装常用操作
- **零外部依赖**: 仅使用项目已有的 requests 和 pydantic

## 安装

如果 SDK 以源码方式集成在仓库中，请将 `libs/atomgit_sdk/src` 加入 `PYTHONPATH`。未来若单独发布，也可直接按包方式安装。

## 快速开始

### 基础使用

```python
from atomgit_sdk import AtomGitClient, AtomGitConfig

# 从配置文件创建
config = AtomGitConfig.from_json("config.json")
client = AtomGitClient(config)

# 获取 PR 信息
pr = client.get_pull_request(123)
print(f"PR Title: {pr['title']}")
```

### 使用 PRService

```python
from atomgit_sdk.services import PRService

service = PRService(client)

# 获取完整的 PR 上下文
context = service.get_full_pr_context(123)
print(f"Files changed: {len(context['files'])}")

# 提交评论
service.submit_inline_comment(123, {
    "path": "src/main.py",
    "position": 10,
    "body": "建议优化这段代码"
})
```

### 回复单条 PR review 意见

官方 API 为 review discussion 提供了独立的回复与解决状态接口。SDK 中应优先使用这些封装，而不是在 skill 脚本里直接拼 HTTP：

```python
from atomgit_sdk.services import RepairService

repair = RepairService(client)

# 按评论 ID 回复；SDK 会先读取该评论并找到 discussion_id
repair.reply_to_comment(123, comment_id=456, reply_body="已按建议修复。")

# 按评论 ID 标记其 discussion 已解决
repair.resolve_comment(123, comment_id=456, resolved=True)

# 已知 discussion_id 时也可以直接调用
client.reply_to_pr_discussion(123, "discussion-id", "补充说明...")
client.set_pr_discussion_resolved(123, "discussion-id", True)
```

### 调用官方 API catalog

当前官方 API 文档大多只公开标题、HTTP 方法和路径。SDK 因此采用“catalog + typed wrapper”的方式：常用协作流程使用显式方法，长尾 API 通过 catalog 调用。

```python
from atomgit_sdk import APICatalog

# 内置常用协作 endpoint
matches = client.find_api_endpoints(path_contains="/pulls/comments")

# 调用内置 catalog 中的 endpoint
comment = client.call_api(
    "get-api-v-5-repos-owner-repo-pulls-comments-id",
    path_params={"owner": "openEuler", "repo": "IB_Robot", "id": 456},
)

# 需要完整官方 endpoint 列表时，可从文档同步（2026-04 抽取到 247 个 API 页面）
full_catalog = APICatalog.from_docs()
```

### Diff 位置计算

```python
from atomgit_sdk.utils import calculate_diff_position

patch = """@@ -10,5 +10,6 @@
 context line
-old line
+new line
 another line"""

position = calculate_diff_position(patch, line_number=11)
print(f"Position in diff: {position}")
```

## 目录结构

```
libs/atomgit_sdk/
├── src/
│   └── atomgit_sdk/
│       ├── __init__.py          # 导出常用类
│       ├── config.py            # 配置管理
│       ├── client.py            # API 客户端
│       ├── models.py            # 数据模型
│       ├── exceptions.py        # 自定义异常
│       ├── utils/               # 工具函数
│       │   ├── diff.py          # Diff 解析
│       │   └── url.py           # URL 解析
│       └── services/            # 业务服务
│           ├── pr_service.py    # PR 操作
│           └── repair_service.py # 修复操作
└── tests/                       # 单元测试
```

## 配置

在你的 `config.json` 中配置：

```json
{
  "atomgit": {
    "token": "your-token",
    "owner": "owner-name",
    "repo": "repo-name",
    "baseUrl": "https://api.atomgit.com"
  }
}
```

## API 参考

### AtomGitConfig

配置管理类。

**方法:**
- `from_json(path: str)`: 从 JSON 文件加载配置

### AtomGitClient

API 客户端。

**方法:**
- `get_pull_request(pr_number: int)`: 获取 PR 详情
- `get_pr_files(pr_number: int)`: 获取 PR 文件列表
- `get_pr_files_json(pr_number: int)`: 获取 PR files.json 列表
- `get_pr_commits(pr_number: int)`: 获取 PR 提交列表
- `get_pr_comments(pr_number: int)`: 获取 PR 评论列表
- `get_pr_comment(comment_id: int)`: 获取单条 PR 评论
- `reply_to_pr_discussion(pr_number: int, discussion_id: str, body: str)`: 回复指定 review discussion
- `set_pr_discussion_resolved(pr_number: int, discussion_id: str, resolved: bool)`: 修改 discussion 解决状态
- `get_file_content(path: str, ref: str)`: 获取文件内容
- `call_api(slug: str, path_params: dict, params: dict, body: dict)`: 调用 catalog 中的官方 API endpoint

### PRService

PR 操作服务。

**方法:**
- `get_full_pr_context(pr_number: int)`: 获取完整 PR 上下文
- `submit_inline_comment(pr_number: int, comment: dict)`: 提交行内评论
- `submit_batch_comments(pr_number: int, comments: list)`: 批量提交评论
- `reply_to_pr_discussion(pr_number: int, discussion_id: str, body: str)`: 回复指定 review discussion
- `set_pr_discussion_resolved(pr_number: int, discussion_id: str, resolved: bool)`: 修改 discussion 解决状态

### RepairService

评审意见处理服务。

**方法:**
- `get_unresolved_comments(pr_number: int)`: 获取未解决 review 评论
- `reply_to_comment(pr_number: int, comment_id: int, reply_body: str)`: 按评论 ID 回复单条 review 意见
- `resolve_comment(pr_number: int, comment_id: int, resolved: bool)`: 按评论 ID 修改所属 discussion 解决状态
- `resolve_discussion(pr_number: int, discussion_id: str, resolved: bool)`: 按 discussion ID 修改解决状态

## 测试

运行单元测试：

```bash
cd libs/atomgit_sdk
pytest tests/
```

## 版本历史

- **0.1.0** (2026-03-20): 初始版本
  - 统一的 AtomGit API 客户端
  - Diff 解析算法（修复版）
  - PRService 和 RepairService
  - Pydantic 数据模型

## 许可证

许可证遵循分发此 SDK 的仓库配置。

# 来源与本地改动

## 上游

- 项目：[mzopedia/simplified-technical-chinese](https://github.com/mzopedia/simplified-technical-chinese)（简明技术中文 / STC）
- 引入版本：v0.2
- 引入 commit：`90ad0f004b26d7a66e286980b5efaf4769ad8584`（2026-10-05，「0.2：在 350 万字真实文档上校准规则、阈值和词表」）
- 引入日期：2026-10-26

## 许可证

- `规范.md`、`词表.md`、`README.md`、`SKILL.md` 的文本：CC BY 4.0（`LICENSE-CC-BY-4.0`），保留上游署名。
- `tools/`、`tests/`、`docs/校准报告.md`：MIT（`LICENSE`），Copyright (c) 2026 mzopedia。

本目录是对上游项目的 vendored 副本，不含 `.git`。按仓库 AI 辅助贡献规范，
涉及本目录的 PR 必须在披露中列出本文件指向的第三方材料与许可证。

## 相对上游的本地改动

只改了 `SKILL.md`，其余文件与上游 v0.2 逐字一致：

1. frontmatter 增加 `license` 字段，指向目录内两份许可证文件。
2. 末尾增加「IB_Robot 仓库约定」一节。该节说明检查范围、调用路径、排除目录和术语边界。

## 目录布局说明

保留上游原布局：`规范.md`、`词表.md` 在根目录，脚本在 `tools/`。
不按本仓库惯例挪到 `scripts/`，原因是 `tools/check.py` 按 `../词表.md` 相对路径读取词表。
保持原布局可以让升级 diff 最小、脚本零改动可用。

## 升级方法

```bash
git clone --depth 1 https://github.com/mzopedia/simplified-technical-chinese.git /tmp/stc-src
# 比对后覆盖除 SKILL.md 外的文件，再把 SKILL.md 的上游改动合入本地版本
diff -ru --exclude=.git /tmp/stc-src .agents/skills/simplified-technical-chinese
```

升级后运行本目录测试确认脚本与词表仍然配套：

```bash
cd .agents/skills/simplified-technical-chinese && python3 -m unittest discover tests
```

## 全局副本

另有一份**未经本地修改**的上游 v0.2 副本安装在用户级目录
`~/.agents/skills/simplified-technical-chinese/`，供未配置仓库 skill 的
协作场景使用。两份内容同源；仓库版多出本文件与 SKILL.md 的本地节。

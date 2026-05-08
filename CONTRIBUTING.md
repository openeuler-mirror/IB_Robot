# 参与贡献 IB_Robot

## 配置个人 Fork 和 Remote

如果你计划参与贡献，需要手动配置个人 fork 和 remote：

### 主仓库

```bash
# 将你的 fork 设置为 'origin'
git remote set-url origin https://gitcode.com/<你的用户名>/IB_Robot.git

# 添加上游仓库
git remote add upstream https://atomgit.com/openeuler/IB_Robot.git
```

### 验证 Remote 配置

```bash
git remote -v
# origin   https://gitcode.com/<你的用户名>/IB_Robot.git (fetch)
# upstream https://atomgit.com/openeuler/IB_Robot.git (fetch)
```

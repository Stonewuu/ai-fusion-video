# GitHub Star 趋势图迁移完成总结

## 最终架构

- Star History 生成器和定时 Pages 工作流归属 `Stonewuu/Stonewuu`。
- `STAR_HISTORY_REPOSITORIES` 控制生成目标；空值默认只生成 `Stonewuu/ai-fusion-video`。
- `STAR_HISTORY_TOKEN` 提供用户身份，用于满足 GitHub 对 Stargazers 列表的管理员或协作者限制。
- 本仓库只保留 README 图片引用，不再运行 Star History Action。

## 固定地址

`https://stonewuu.github.io/Stonewuu/star-history/Stonewuu/ai-fusion-video.svg`

## 构建隔离

- `.github/workflows/docker-publish.yml` 未修改。
- 本仓库删除 `.github/workflows/star-history-pages.yml`。
- Tag push 不会触发 Profile 仓库工作流，Profile 工作流也不会访问本仓库构建、发布或 Docker Secrets。

## 验证

- Profile Node.js 单元测试：4/4 通过。
- 使用仓库所有者令牌真实读取 1,218 条 Stargazer 记录：通过。
- SVG XML 与浏览器渲染：通过。
- Profile Star History 工作流 Actionlint：通过。
- 本仓库 Docker 工作流 diff：空。
- 双仓库 `git diff --check`：通过。

## 发布

- `Stonewuu/Stonewuu`：提交 `a3079fa`，草稿 PR `#1`。
- `Stonewuu/ai-fusion-video`：提交 `bb65635`，草稿 PR `#53`。

## 用户配置

1. 在 Profile 仓库添加 Secret `STAR_HISTORY_TOKEN`。
2. 可选添加 Variable `STAR_HISTORY_REPOSITORIES`。
3. 将 Profile 仓库 Pages Source 设置为 `GitHub Actions`。
4. 合并 Profile PR 后手动运行一次 `Generate Repository Star History`。

# GitHub Star 趋势图迁移完成总结

## 最终架构

- Star History 生成器和定时 Pages 工作流归属 `Stonewuu/Stonewuu`。
- `STAR_HISTORY_REPOSITORIES` 控制公开仓库列表；空值默认只生成 `Stonewuu/ai-fusion-video`。
- 工作流只读取公开仓库的 `stargazers_count`，不调用受限的 Stargazers 列表端点，不需要 PAT 或自定义 Secret。
- `ai-fusion-video` 的历史曲线以 110 个日期点的种子迁入 Profile 仓库；后续历史状态随 Pages JSON 部署持久化，不产生生成物提交。
- 新增仓库从首次成功运行开始记录每日净 Star 数。
- 本仓库只保留 README 图片引用，不包含 Star History Action、生成器或测试。

## 固定地址

`https://stonewuu.github.io/Stonewuu/star-history/Stonewuu/ai-fusion-video.svg`

## 构建与发布隔离

- 本仓库已删除 `.github/workflows/star-history-pages.yml`。
- `.github/workflows/docker-publish.yml` 迁移前后 Git blob 均为 `6d60a2a0bf66908b0f0de2d13e1526598d24c2bf`。
- `Publish Docker Images` 保持 active，Tag push 触发器、Docker Secrets、权限和构建步骤未改变。
- Profile 工作流只监听 `schedule` 与 `workflow_dispatch`，不会由本仓库 Tag push 触发，也不会访问本仓库构建或发布 Secrets。

## 验证结果

- Profile Node.js 单元测试：6/6 通过。
- 无 PAT 本地真实 API 生成：1,218 Stars、110 个历史日期点。
- Profile 工作流首次默认部署：运行 `30925998858` 成功。
- 临时配置两个仓库：运行 `30926160566` 成功，两份 JSON 均返回 200。
- 删除变量后恢复默认：运行 `30926253788` 成功；`ai-fusion-video` 返回 200，额外仓库返回 404，首页只列出默认仓库。
- SVG 公开响应为 `200 image/svg+xml`，历史 JSON 为 200 且包含 110 个日期点。
- SVG 浏览器渲染、Actionlint、双仓库 `git diff --check` 均通过。

## 发布结果

- `Stonewuu/Stonewuu` PR `#1` 已合并，merge commit 为 `eb1e8567e71d8ccc418f522511762aece7c04525`。
- `Stonewuu/ai-fusion-video` PR `#53` 已合并，merge commit 为 `ecd5a03ba8c4aa1768bcdd0da7a4bb7e2bc599d4`。
- Profile Pages 已启用，Source 为 `GitHub Actions`。

## 用户控制

不需要配置 Secret。可在 `Stonewuu/Stonewuu` 的 Actions Variables 中设置 `STAR_HISTORY_REPOSITORIES`，使用逗号、空格或换行分隔 `owner/repository`；删除或留空该变量时恢复默认仓库。

## 遗留事项

无。

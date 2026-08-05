# GitHub Star 趋势图迁移进度

## 完成状态

- [x] 确认仓库内置 `GITHUB_TOKEN` 被受限 Stargazers 列表端点拒绝。
- [x] 改用公开 `stargazers_count` 定时采样，不需要 PAT。
- [x] 将 `ai-fusion-video` 的现有历史曲线迁移为 Profile 仓库种子。
- [x] 在 Profile 仓库实现可配置多仓库生成器和独立 Pages 工作流。
- [x] 实测变量配置两个仓库，并实测删除变量恢复默认仓库。
- [x] 从本仓库删除 Star History 工作流、生成器和测试。
- [x] 更新中英文 README 固定地址。
- [x] 验证 Docker Tag 发布工作流 blob 未变化且保持 active。
- [x] 合并双仓库 PR，启用 Profile Pages 并完成三次真实部署。

## 最终状态

- Profile Pages：`https://stonewuu.github.io/Stonewuu/`
- 默认 SVG：`https://stonewuu.github.io/Stonewuu/star-history/Stonewuu/ai-fusion-video.svg`
- Profile PR：`Stonewuu/Stonewuu#1`，已合并。
- 项目 PR：`Stonewuu/ai-fusion-video#53`，已合并。
- 默认变量状态：`STAR_HISTORY_REPOSITORIES` 未设置，仅生成 `Stonewuu/ai-fusion-video`。

# GitHub Star 趋势图迁移计划

## 目标

- 将 Star History 生成与 Pages 部署完整迁移到 `Stonewuu/Stonewuu`。
- 本仓库只保留固定图片引用，不包含任何 Star History Action 或生成脚本。
- 不修改 `.github/workflows/docker-publish.yml`，确保 Tag 发版路径不受影响。
- 由 Profile 仓库变量控制需要生成的仓库列表，未配置时只生成 `Stonewuu/ai-fusion-video`。

## 验收标准

- 本仓库 Star History 工作流、脚本和测试全部删除。
- 中英文 README 指向 Profile Pages 新地址。
- Docker Tag 发布工作流内容与迁移前完全一致。
- Profile 仓库脚本、测试和工作流通过验证。

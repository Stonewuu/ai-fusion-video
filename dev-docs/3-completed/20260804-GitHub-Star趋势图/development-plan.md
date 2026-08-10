# GitHub Star 趋势图开发计划

## 目标

- 使用仓库内置 `GITHUB_TOKEN` 读取当前仓库 Stargazers 历史。
- 每 6 小时生成一次静态 SVG 趋势图。
- 通过 GitHub Pages artifact 部署到固定 URL，不产生定时 Git 提交。
- README 与 README_EN 使用同一个固定地址。

## 实现范围

- 新增无第三方依赖的 Node.js SVG 生成脚本。
- 新增脚本单元测试。
- 新增 GitHub Pages 定时部署工作流。
- 记录首次启用 GitHub Pages 的一次性配置步骤。

## 验收标准

- 本地单元测试通过。
- 使用仓库所有者令牌可读取完整 Stargazers 列表并生成有效 SVG。
- 工作流语法通过静态检查。
- 周期运行只创建 Pages deployment，不修改仓库内容或产生提交。

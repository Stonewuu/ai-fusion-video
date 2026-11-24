# v1.4.2 发布与远程升级完成记录

2026 年 11 月 25 日已完成全部代码提交、版本分支合并、正式版本发布和远程升级。

## 发布范围

包含 Agent 消息持久化与 Redis 流缓存优化、智能助手及 Pipeline 历史重建、24 小时缓存过期、多子 Agent 顺序与状态修复、模型配置界面调整、图片预览背景和 Docker 密码找回日志修复。具体用户可见变化见同目录的 `release-notes.md`。

- 功能分支：`feat/agent-redis-stream`，代码与测试提交至 `cc12488`。
- 版本分支：从 `v-1.4.1` 创建 `v-1.4.2`，通过合并提交 `f5945a5` 纳入功能分支，两个分支均已推送。
- 发布标签：`v1.4.2`，指向 `2c0fc3f89607d251b21655163593e38a7a0f9317`；前后端产品版本均为 1.4.2。
- [正式 Release](https://github.com/Stonewuu/ai-fusion-video/releases/tag/v1.4.2) 于北京时间 01:01 公开，并设为最新正式版本。
- [镜像构建](https://github.com/Stonewuu/ai-fusion-video/actions/runs/45439113692) 成功，前后端均提供 `linux/amd64` 和 `linux/arm64` 镜像，`1.4.2` 与 `latest` 的摘要一致。

## 验证结果

| 范围 | 结果 |
| --- | --- |
| 前端回归 | 309 项脚本测试通过，TypeScript 检查通过 |
| 前端代码检查与构建 | ESLint 0 错误、6 条既有警告；独立目录生产构建及发布流水线构建通过 |
| 后端回归 | 常规测试共 1,212 项，无失败、无错误，1 项媒体测试组按启用条件跳过；随后启用该组，4 项测试通过 |
| 发布产物 | 前后端两个架构构建成功；远程 ARM64 镜像摘要与发布产物一致 |
| 远程服务 | 后端健康检查通过，前端运行正常，数据库迁移成功 |
| 外网与页面 | HTTPS 页面返回 200，初始化接口正常；Chrome 登录后显示当前版本和最新可部署版本均为 v1.4.2 |

## 远程升级

[融光网站](https://afv.stonewu.com) 使用 `/opt/programs/ai-fusion-video` 的 Compose 部署。升级前确认无活动 Agent 任务，并备份应用数据库与部署配置，最终备份目录为 `/opt/programs/ai-fusion-video/backups/pre-1.4.2-20261125-005812`。

前后端已分别切换到 `stonewuu/ai-fusion-video:1.4.2` 和 `stonewuu/ai-fusion-video-web:1.4.2`。应用网关已重新加载；数据库、Redis、媒体与 Agent 工作目录的数据和挂载保持原样，共享数据库及其他项目服务未重启。

Flyway 已成功执行 `V1.1.4.2.0__agent_stream_protocol.sql`。旧版本镜像及升级前备份保留。用户刷新网页即可使用新版本。

## 后续事项

本次发布与远程升级没有未完成项。构建工作流的运行时升级提示记录在 `technical-debt.md`，发布记录后续提交到版本分支，不移动已发布标签。

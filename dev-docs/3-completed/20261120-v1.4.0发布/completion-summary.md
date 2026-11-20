# v1.4.0 发布结果

融光 v1.4.0 于北京时间 2026 年 11 月 20 日 20:32 正式发布。版本标签指向 `v-1.4.0` 分支的 `ce13da81e3934ef74e6d94b3b3c4f103e5443a7c`。GitHub 已设为最新版本，公开正文与仓库中的更新说明一致。[GitHub Release](https://github.com/Stonewuu/ai-fusion-video/releases/tag/v1.4.0)

## 发布范围

本版包含长篇剧本分集导入和解析、剧本与分镜的独立清空和重新关联、长目录与资产编辑、助手长消息恢复、APIXO 接入，以及 ComfyUI 可选媒体输入。发布前已将工作区中的相关源码、模型预设、测试和交付文档提交到版本分支。

## 验证

- 前端生产构建和 TypeScript 检查通过，282 项脚本测试通过。ESLint 报告 6 条警告，0 个错误。
- 后端常规测试报告 1,187 项，其中 FFmpeg 用例按默认配置跳过，其余通过。单独启用 FFmpeg 后，4 组音视频时长组合通过。
- 85 项集成测试通过，覆盖提供商配置、资源读写、历史恢复、外部工具结果及流式回复重连。Flyway 校验通过，已将本版迁移纳入已发布脚本的校验清单。

## Docker 镜像

GitHub Actions 已构建并推送前后端镜像。两个仓库均包含 `linux/amd64` 和 `linux/arm64`，各自的 `1.4.0` 与 `latest` 在发布时指向同一摘要。[构建记录](https://github.com/Stonewuu/ai-fusion-video/actions/runs/44733870571)

| 镜像 | 多架构镜像摘要 |
| --- | --- |
| `stonewuu/ai-fusion-video:1.4.0` | `sha256:a2c1fae630e6d09eb72f8580fd14bc4f4ff97bdde8743c1d1bedafe6abee7e6d` |
| `stonewuu/ai-fusion-video-web:1.4.0` | `sha256:1009835edf51d0decb5f2c90c54463e3138fea63914e3cfa39d8955a67935404` |

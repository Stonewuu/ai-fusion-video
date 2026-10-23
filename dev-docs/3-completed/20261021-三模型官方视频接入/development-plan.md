# Seedance 2.5 / MiniMax H3 / Wan 3.0 官方接入

## 目标与范围

核对三家官方 API 的模型标识、输入组合、尺寸、时长、音频及任务生命周期，完善模型预设、官方供应商协议、前后端参数限制和错误处理。普通视频生成、画布与智能体必须经过同一套能力校验和可靠提交链路。

## 实施顺序

1. 官方资料核对：创建、查询、取消、模型列表与输入约束；记录 URL、核对日期和相互矛盾的说明。
2. 预设与注册：Seedance 2.5、MiniMax H3/H3 Max、Wan 3.0/Prime；MiniMax 官方平台和视频协议，Wan 业务空间 URL 引导。
3. 正式协议：正确传递多模态内容与可选参数，不静默丢弃参数，不自动重复提交；查询、失败原因、取消与媒体接收沿用持久执行链路。
4. 能力校验：输入互斥、数量总限、时长/比例随输入变化、音频与种子支持、自动时长，以及模型确实支持的高级参数。
5. 验证：独立协议测试、真实 HTTP 模拟、能力与预设测试、前端参数回归、构建和 Chrome 配置验收。真实供应商调用需有对应官方配置及明确费用授权，不以本机模拟冒充真实生成。

不修改已发布迁移；预设优先使用已有 classpath JSON 机制。若确需数据库结构修改，先按 AGENTS.md 核对正式发布边界与目标库。

## 官方依据（2026-10-21）

- [火山方舟模型列表](https://docs.volcengine.com/docs/ark/model-list?lang=zh)
- [Seedance 2.5 教程](https://docs.volcengine.com/docs/ark/seedance-2-5?lang=zh)
- [创建视频任务](https://docs.volcengine.com/docs/ark/create-video-generation-task-api?lang=zh)
- [MiniMax H3 创建任务](https://platform.minimax.io/docs/api-reference/video-generation-v2-create)
- [MiniMax H3 查询任务](https://platform.minimax.io/docs/api-reference/video-generation-v2-query)
- [MiniMax H3 取消/删除任务](https://platform.minimax.io/docs/api-reference/video-generation-v2-delete)
- [Wan 3.0 API](https://www.alibabacloud.com/help/zh/model-studio/wan3-video-generation-api-reference)
- [Wan 3.0 使用指南](https://www.alibabacloud.com/help/zh/model-studio/wan3-video-generation-guide)

模型 ID：`doubao-seedance-2-5-260628`、`MiniMax-H3`、`MiniMax-H3-Max`、`wan3.0-video`、`wan3.0-video-prime`。不以第三方聚合平台的同名接口替代官方协议。

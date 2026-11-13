# Gemini 官方模型对应表

核对日期：2026-11-06。按完整模型 ID 对应，来源链接随预设保存。生命周期和可用地域以对应渠道为准。

| 模型 ID | 类型 | 上下文 | 单次最大输出 | 官方说明 |
| --- | --- | ---: | ---: | --- |
| gemini-4-argon | 对话 | 1048576 | 65536 | [模型说明](https://aistudio.google.com/docs/models/gemini-4-argon) |
| gemini-3.8-flash | 对话 | 1048576 | 65536 | [模型说明](https://aistudio.google.com/docs/models/gemini-3.8-flash) |
| gemini-3.7-flash | 对话 | 1048576 | 65536 | [模型说明](https://aistudio.google.com/docs/models/gemini-3.7-flash) |
| gemini-3.6-flash | 对话 | 1048576 | 65536 | [模型说明](https://aistudio.google.com/docs/models/gemini-3.6-flash) |
| gemini-3.5-flash | 对话 | 1048576 | 65536 | [模型说明](https://aistudio.google.com/docs/models/gemini-3.5-flash) |
| gemini-3.5-flash-lite | 对话 | 1048576 | 65536 | [模型说明](https://aistudio.google.com/docs/models/gemini-3.5-flash-lite) |
| gemini-3.1-flash-lite | 对话 | 1048576 | 65536 | [模型说明](https://aistudio.google.com/docs/models/gemini-3.1-flash-lite) |
| gemini-3.1-pro-preview | 对话 | 1048576 | 65536 | [模型说明](https://aistudio.google.com/docs/models/gemini-3.1-pro-preview) |
| gemini-3-flash-preview | 对话 | 1048576 | 65536 | [模型说明](https://aistudio.google.com/docs/models/gemini-3-flash-preview) |
| gemini-2.5-pro | 对话 | 1048576 | 65536 | [模型说明](https://aistudio.google.com/docs/models/gemini-2.5-pro) |
| gemini-2.5-flash | 对话 | 1048576 | 65536 | [模型说明](https://aistudio.google.com/docs/models/gemini-2.5-flash) |
| gemini-2.5-flash-lite | 对话 | 1048576 | 65536 | [模型说明](https://aistudio.google.com/docs/models/gemini-2.5-flash-lite) |
| gemini-robotics-er-2-preview | 对话 | 131072 | 65536 | [模型说明](https://aistudio.google.com/docs/models/gemini-robotics-er-2-preview) |
| gemini-3.1-pro-preview-customtools | 对话 | 1048576 | 65536 | [模型说明](https://aistudio.google.com/docs/models/gemini-3.1-pro-preview) |
| gemini-nano-banana-2.1 | 图片 | 131072 | 32768 | [模型说明](https://aistudio.google.com/docs/models/gemini-nano-banana-2.1) |
| gemini-3.1-flash-image | 图片 | 131072 | 32768 | [模型说明](https://aistudio.google.com/docs/models/gemini-3.1-flash-image) |
| gemini-3.1-flash-lite-image | 图片 | 65536 | 4096 | [模型说明](https://aistudio.google.com/docs/models/gemini-3.1-flash-lite-image) |
| gemini-3-pro-image | 图片 | 65536 | 32768 | [模型说明](https://aistudio.google.com/docs/models/gemini-3-pro-image) |
| gemini-2.5-flash-image | 图片 | 65536 | 32768 | [模型说明](https://aistudio.google.com/docs/models/gemini-2.5-flash-image) |

## 其他目录项

| 官方类型／型号 | 本项目状态 |
| --- | --- |
| Gemini Omni 1.1 / 1.2 Flash、Omni 预览版 | 视频接口尚未接入，不再归入对话模型。Models.dev 可按视频类型检索能力。 |
| Gemini 3.8 Live / Extended Thinking、3.1 Flash Live、2.5 Flash Native Audio、3.5 Live Translate、Robotics ER 2 Streaming | 实时双向接口尚未接入，不创建普通对话预设。 |
| Gemini 3.8 Flash / Flash-Lite TTS、3.1 Flash TTS、2.5 Flash / Pro TTS | 语音合成接口尚未接入。 |
| Gemini Embedding 001 / 2、3.5 Transcribe / Transcribe Live | 嵌入与转写接口尚未接入。 |
| Gemini Deep Research / Deep Research Max | 使用托管智能体接口，本轮没有用普通聊天接口代替。 |
| Gemini 2.0 Flash / Lite、3 Pro Preview、3.1 Flash-Lite Preview、Robotics ER 1.6、2.5 Computer Use | 官方目录标记已停用，不新增可执行预设。 |

Gemini 3.7 Flash 与 3.5 Flash 在 Developer API 中属于官方自动路由的历史名称；Vertex 生命周期独立。2.5 Flash / Pro 在 Developer API 中已弃用但尚未到文档公布的停止日期，预设保留完整旧型号。

Flow 图片预设 4 项保留原标识与能力数据，名称增加渠道标识，官方 Google 图片预设使用 google: 前缀避免冲突。

官方目录：[Gemini Developer API](https://aistudio.google.com/docs/models)、[Vertex 模型目录](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/google-models)、[停用时间表](https://aistudio.google.com/docs/deprecations)。

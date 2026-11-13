# 实现边界与后续事项

## 当前边界

- 只维护文字、图像、视频、音频生成。已删除被取消类型的组件、执行器与预设，不保留通用“模型工具”接口或隐藏菜单。
- 历史迁移 1.1.3.1.1 已在保留数据的本地库执行，因此新增 1.1.3.1.2 转换任务表和音乐类型，不回改历史脚本。旧配置停用、旧任务归档，已生成媒体和引用继续保留。无需为旧任务接口增加运行时兼容代码。
- Lyria 002 的官方文档列出 audioContent/mimeType，本账号返回 bytesBase64Encoded WAV。仅支持这两个已有依据的结构；其他结构明确失败，协议回归保护这项差异。
- 音频没有可恢复的供应商回执时，连接中断标为结果待确认，不自动重复生成。已收到的完整输出可在保存失败后重试；供应商没有返回音频时不能重试保存。
- Lyria RealTime 只收集音乐流并保存为音频文件，不提供实时对话、翻译、转写或麦克风会话。

## 尚未验证通过

- Lyria 3 Clip 的 Vertex 响应缺少音频。官方最小请求对照也为空；暂未确定是接口行为还是当前账号的可用性问题，不断言已经排除所有对接因素。保留原始响应和明确失败状态，后续可与 Google 支持核对。
- Veo 3.1 Fast 返回服务繁忙；Gemini Developer API 无独立凭据。当前没有音频或视频成功证据的型号不记为生成成功。

## 官方依据

- [Vertex Interactions 概览](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/interactions/)：Lyria 3 只支持 store=false。
- [Lyria 音乐生成](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/music/generate-music)：请求步骤和音频输出结构。
- [Gemini 语音生成](https://aistudio.google.com/docs/speech-generation)：TTS 请求与返回格式。

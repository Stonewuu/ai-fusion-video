# 边界与待确认事项

- Google Cloud API Key、AI Studio API Key、服务账号 JSON 属于不同接入方式，必须按选择明确校验，不能在失败后自动改用另一套凭据。
- 模型目录可列出不适用于当前地域、未获项目授权或本项目尚不支持的模型；远程发现不能宣称每个目录条目都已通过生成测试。
- 专用协议归属由提供商目录的 restrictedProtocols 声明，按 API 配置的 providerId 校验；不根据名称、URL 或赞助展示状态猜测渠道身份。
- 云端密钥与服务账号私钥不进入日志、截图、测试报告或 Git；验收附件统一放入 .local/verification。

## 已确认的适用边界

- Google 官方 Model Garden 目录当前拒绝 API Key，要求 OAuth。API Key 模式不自动挪用已保存的服务账号凭据；用户可以显式切换服务账号获取模型，再切回 API Key 调用。这是上游鉴权限制，不是未完成的列表实现。
- 自动发现仅接入 Google 发布者下本项目支持的对话和图片模型，不导入嵌入、实时语音、转写、TTS 和 Veo；目录出现某模型不代表当前项目在所有区域都可调用。
- Gemini 图片模型使用 generateContent，支持参考图和两种鉴权；Imagen predict 保持服务账号方式，API Key 选择 Imagen 会得到明确提示。本轮对 Gemini 两种鉴权进行了外部生成验证，没有进行 Imagen 付费生成。
- 服务账号留空时保留原有服务器默认凭据（ADC）用法；本机使用显式 JSON 验证。旧 Imagen 自定义 predict 地址及早期字段兼容逻辑未在此次改造中扩展。
- 无新增数据库迁移，无硬编码模型目录、无静默鉴权或协议替换。

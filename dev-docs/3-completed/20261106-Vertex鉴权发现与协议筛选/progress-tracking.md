# 当前进度

- 已确认当前 Vertex 自动发现直接抛出“尚未接入”，不会发出远程请求。
- 当前配置 ID 8 使用服务账号 JSON，JSON 类型与必需字段完整，项目 ID 与用户打开的 Google Cloud 项目一致。
- 当前模型 ID 18 为 Gemini 3.8 Flash，默认地域为 us-central1。Java 的 Vertex SDK、OAuth 取令牌和图片调用均已有显式 API 代理接入点，仍需测试实际连通性。
- 本机 mihomo 监听 7890，当前进程环境有 HTTP_PROXY、HTTPS_PROXY、ALL_PROXY，不能仅凭这些变量推断 Java 请求走了代理或 TUN。
- Google Cloud 控制台中已有绑定服务账号的 Vertex API 授权密钥，可用于 API Key 方式验证。只查看了现有配置，没有创建密钥或修改权限。
- 已用同一份服务账号分别测试 Java 不显式设代理与显式 Clash 7890：OAuth 成功，目录接口均返回 200、33 项；gemini-3.8-flash 在 us-central1 返回 404，在 global 的 countTokens 返回 200。因此当前故障主因是模型地域，不是 JSON 格式或本机 Java 完全无法连接 Google。
- 现有 Google Cloud API Key 在 Express 全球端点的 countTokens 返回 200；同一密钥查询 Model Garden 目录返回 401，明确要求 OAuth。没有创建凭据、修改 IAM 或放宽密钥限制。
- 已实现显式服务账号／API Key 接入类型、官方目录分页、Gemini 图片 generateContent 请求，以及前后端按渠道限制专用协议。API Key 模式不会改用已保存的 JSON。
- 85 项后端定向测试、1 项 MySQL 持久化集成测试和 9 项前端测试通过，TypeScript 与修改文件 ESLint 通过。后端打包完成并已重启，启动时校验通过 35 条 Flyway 迁移。
- Chrome 中服务账号模式获取 22 个对话／图片模型，导入 Gemini 3.1 Flash Image 成功；两种鉴权下 Gemini 3.8 Flash 均返回 OK。
- API Key 模式生成蓝色杯子；服务账号模式用这张图片作参考，生成绿色版本。两次任务均为 SUCCEEDED / FINISHED，结果自动保存到平台媒体记录。
- 已验证 API 配置和模型覆盖选项：普通渠道没有秘塔／无限星河专用协议，无限星河只能选择自己的专用协议；切换为普通渠道会清除原专用覆盖。旧秘塔配置已通过页面补齐 providerId。
- 当前 Vertex 配置保留服务账号 + global，已有 API Key 同时保留，用户可以自行切换；原默认文本和图片模型未更换。Chrome 已检查桌面、390px 窄屏、明暗主题及表单焦点／必填错误状态，视口和浅色主题已恢复。

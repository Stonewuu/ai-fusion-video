## 2026-10-15 阶段一：确认流式阻塞来源

- 已按规则先使用两仓 CodeGraph；参考代码实际位于 `agenteam-package/agenteam`。
- 当前 `AgentScopeModelFactory.provisionRequest` 调用 `provisionSingleRequestModel`；`SingleRequestModelSupport.options` 强制 `stream(false)`。这是模型端先等完整回复的直接原因。
- `TrackedAgentChatModel.doStream` 在 `Mono.usingWhen` 内 `collectList()`，再 `flatMapMany` 发送。即使提供商真的产生增量，这层仍会全部收集后一次输出。
- 当前 `AgentRunReplayService` 以 MySQL 日志作为实时读取源，每轮读取快照及事件页，Redis 只用于唤醒，300ms 轮询；需继续核实写入、消息保存和子事件镜像对输出的阻塞。
- agenteam 的 `FrameEventPersistence` 每 50ms 合并增量；`LiveExecutionSession` 分离实时发送与完整内容块保存队列；Redis 保存活动全文及近期事件，快照组合历史与活动内容并携带一致读取位置。
- 当前 `AgentScopeV2Properties.Execution` 默认运行 30 分钟、999 步；`AgentScopeSubAgentToolAdapter` 继承父截止时间并显式 `timeout`。
- 尚未修改业务代码。

下一步：查清单次提交提供商传输、执行限制/租约、事件提交链路及前端恢复；建立能在完整回复结束前断言首段到达的回归测试，然后按链路实施重构。

## 2026-10-15 阶段二：模型真实流式与执行限制

- 已将 `SingleRequestModelSupport`、OpenAI、DashScope 改为流式，保留单次 HTTP 提交与禁止隐藏重试。
- Anthropic 改为 `createStreaming`，新增按内容块 index 保留工具身份、完成原因及累计用量的解析；旧 SDK 解析器会过滤掉无正文的用量片段。
- Gemini/Vertex 的 Google GenAI 1.45 `ResponseStream.next()` 会预读下一段才返回当前段，真实 HTTP 首段测试确实超时。新增 `com.google.genai.GenerationContentStream`，复用 SDK 的请求/响应转换和既有鉴权传输，按 SSE 事件边界即时返回，取消时关闭原响应。该适配与现有 `GenerationHttpApiClient` 同处 SDK 包；不使用反射或重发。
- Ollama 改为原协议流式读取。OpenAI 新增 `StreamingOpenAiFormatter`，按工具 index 还原 id/name，避免并行工具的后续片段误入最后一个工具。
- `TrackedAgentChatModel` 改为 `Flux.usingWhen`；不再 `collectList`；正文和思考立即通过，只累计工具参数，在流完成前校验完整参数。无完成标记的 EOF 登记为结果不确定；已正常完成但参数无效时仍登记真实用量。
- `SingleRequestTextProvidersTests`、`TrackedAgentChatModelTests`、`TrackedAgentStreamingTests` 已在 2.0.0 下全部通过（52 个测试）；包括七协议真实 HTTP 门闩测试：服务器在首段被消费者读到前不发送下一段，证明并非最终内容正确但仍整段缓冲。
- 新增 `AgentExecutionLimits`，配置 0 表示不限时/不限步数。既有数据库 deadline_at 为 NOT NULL，使用 MySQL 最大日期作为明确无期限编码；遇到此编码不建立运行计时器，也不给工具/子任务附加 Reactor timeout。SDK 仅支持正整数步数，按 agenteam 转换为 Integer.MAX_VALUE；SDK 工具默认五分钟通过最大可表示等待时间覆盖（与 agenteam 相同）。保留租约、取消、显式有限配置及审批有效期。
- 已修改普通/画布 agent 启动、审批后恢复、画布外部等待恢复和 SDK toolExecutionConfig。默认 YAML 改成 run-timeout=0s/max-iters=0。
- 限制相关初轮 45 个测试有 44 个通过，仅 MCP 过期错误文案断言因共享期限检查变更失败，已更新该断言，待重新运行。
- 用户补充要求升级；已查证参考项目 AgentScope 为 2.0.4，已改 pom 并启动编译。当前 Maven 编译执行句柄为 1848（需实际查询状态，不能凭此记录认为仍运行）。

下一步：完成 AgentScope 2.0.4 编译/API 适配与模型测试重跑；补不限时的虚拟时间/取消回归测试；然后继续数据库实时发送、恢复与前端链路重构。当前尚不能宣称整个目标完成。

## 2026-10-15 阶段三：2.0.4 接入与客户端恢复

- AgentScope 2.0.4 编译通过；七协议流式、单次提交、用量、工具完整性、模型内核、运行控制和技能目录相关测试已重新通过。技能目录测试改用新版显式 RuntimeContext 接口。
- 已核对 2.0.4 `ReActAgent` 源码：工具挂起会原生发出 `RequireExternalExecutionEvent` 并保存对应 replyId。删除 `AgentExecutionFactory` 中为 2.0.0 手工补事件的旧逻辑，避免产生两个不一致的等待候选。
- 新增 SSE 心跳（15 秒、无业务 id）；只对 Pipeline SSE 设置 no-transform/X-Accel-Buffering=no，并取消 Servlet 默认异步连接超时。心跳不会重复订阅执行流，终态或错误后退出；相关测试通过。
- `AgentExecutionLimitsTests` 验证默认不限时/步数、无运行计时器、虚拟推进 100 年仍未结束、用户取消仍可释放执行、子任务等待三天可完成。
- 新增前端逐行 SSE 解码器，正确处理 CRLF 与中文 UTF-8 跨网络分片，不再把未结束的终态事件当成完成；收到根终态立即结束读取，不等代理关闭连接；消费者抛错会关闭读取连接。
- 助手连接协调器移除无限增长的重复事件集合，仅使用成功应用事件后的序号；归约失败向传输层抛错，游标不前移。第一次响应前断线不再擅自视为业务失败，也不会重新 POST 启动。
- 修复断线期间完成却只保留半段文字：远端最后序号较新时重新加载完整历史。切换运行时清空旧轮次时间线，防止串轮。历史请求跨越实时终态/下一轮时会重新读取，不覆盖更新的实时正文。
- 前端 `pipeline-streaming.test.mjs` 与原 `assistant-restoration.test.mjs` 共 16 项通过；TypeScript 和相关 ESLint 通过。
- `AgentReplayLiveIT` 的真实 MySQL/Redis 四项通过；`AgentWaitingStateIT` 七项通过（已按新默认值检验无期限恢复，并保留审批过期及并发恢复断言）。
- 原 `PlatformSubAgentRunServiceIT` 缺少独立 Redis 配置，被现有本地开发服务保护拦截；已给测试添加独立容器并补齐 publishWakeup 的 mock。未开始执行的子任务可立即 CANCELLED，断言接受取消已完成或已请求两种合法状态。
- 新增完整 HTTP 集成验收 `AgentStreamingAcceptanceIT` 与本机分片供应商 `AgentStreamingProvider`，真实调用模型适配、AgentScope、日志和 SSE；主 agent 首段门闩、断开后继续、按序号接续、刷新全文一致和默认无期限已通过首轮前半段。子 agent 场景首次因测试缺少 scriptId 被服务端拒绝，已补真实项目/剧本和上下文，正在重跑。
- 正在运行的 Maven 句柄：50779，日志 `ai-fusion-video/target/agent-streaming-acceptance.log`，必须查询实际状态后决定下一步。
- 已准备隔离前端副本 `ai-fusion-video/target/agent-streaming-web`，依赖通过 Junction 复用；尚未启动。用于 Chrome 验收时避免覆盖已有 3000/3023 与 18092 预览进程。
- `PlatformSubAgentRunService` 刚改为只累计正文与终态，不再保存全部事件副本；保留纯空格分片，并忽略孙任务镜像终态。此最新修改待针对性回归。

下一步：完成完整 HTTP 子 agent 验收，记录实际首段延迟；根据实测继续解决日志链路延迟/还原问题；启动专用测试后端和隔离前端，仅使用 Chrome 验证持续输出、刷新、切会话、取消。补长任务与慢存储证据，最终完成全部验证后再归档。

## 2026-10-15 阶段四：真实链路、Chrome 与通知阻塞

- 完整 HTTP 根/子 agent 验收通过：真实 2.0.4、真实工具委派定义、MySQL/Redis 和本地分片模型。最新记录首段到客户端 195ms，连续输出事件到达延迟 P95 117ms；根事件回放 23 条、父子事件 48 条，供应商请求严格 4 次（独立根任务一次，父→子→父三次）。
- 确认并修复另一条阻塞链：原 `appendRequired`、终态提交、审批恢复都会等待 Redis PUBLISH；Redis 等待会让后续已生成内容卡在日志入口。现在只登记待通知的最新序号，由独立 25ms 批次异步发送；发送有容量及并发限制，通知等待不阻塞写入和结束。原始事件仍完整写入 MySQL，未引入会丢失未保存正文的临时显示协议，也未改变工具定义。
- `AgentRunReplayService` 首次读取不再等待 Redis 订阅握手，继续沿同一 MySQL 序号进行周期读取与通知唤醒。`AgentRealtimeDeliveryTests` 以永不完成的 Redis 握手/发送验证首批历史与生产者仍可继续。
- Chrome 会话使用明确 chrome 后端。初次导航超时来自隔离前端首次编译（约 40 秒），编译完成后恢复；全程未使用内置浏览器。浏览器密码管理器提示条遮挡新会话按钮时，通过键盘操作同一个可见按钮完成，不修改应用代码。
- Chrome 观察到首段、第 1 段及停止按钮同时出现，此时未出现后续 19 段；立即刷新后 20 段全部恢复，首段/末段各出现一次。另一轮输出至第 2 段时点击停止，最终保留至第 8 段，会话显示已取消且输入恢复。切换到完整会话再切回，完整会话末段仍只有一次，取消会话没有第 20 段串入。
- 浏览器验收后测试库统计 7 次模型请求（最初 4 次集成 + 3 次浏览器操作），6 次完成、1 次取消后结果待核实，刷新/切换没有额外模型请求。6 个运行中 5 个完成、1 个取消。
- Chrome 复核通过。测试前端 62689 与测试后端 49623 已正常停止，专用 Testcontainers 已随测试退出；未停止或重建用户原有 3000/3023/18092 服务。
- 补充 Responses 协议取消时主动关闭流/客户端，单次模型包装器透传上下文窗口及结构化输出能力；七协议取消分片回归已加入全量测试。
- 后端全量 `mvn test` 执行 750 项：1 项环境条件跳过，其余仅依赖契约测试仍写死 2.0.0 而失败。已按用户要求更新为 2.0.4，正在定向复测并重跑最终回放/等待集成（句柄 18228；日志 `target/agent-streaming-final-replay.log`）。
- 前端类型检查、相关 ESLint 再次通过；流读取、恢复、画布结果关联共 22 项通过。新增历史请求跨过终态的回归，防止旧历史返回后覆盖更晚的实时正文。
- 新增子任务输出测试，验证保留纯空格/换行片段、忽略孙任务镜像终态；待最终定向测试。

下一步：查询句柄 18228 的最终结果，运行子任务输出新增测试，整理当前测试报告和差异；在最终实现上核验主/子 agent 流式、取消、恢复及版本/工具定义约束，完成审计后归档文档。

## 2026-10-15 最终核验

- 18228、17340、8369、94643、86010 均已完成且退出码 0。最终内核关闭额外的 `SessionTranscriptWriter` 与正文追踪日志后，真实 SDK 子任务挂起→释放父执行→接收持久化子结果→父任务恢复，以及取消后新消息恢复历史两项再次通过。
- 新增模拟慢日志测试：模型每 100ms 输出，合并等待 50ms、每次写入延迟 80ms；首条在整轮完成前提交，后续 12 条保持顺序并独立推进，没有整轮收集后批量交付。
- 子任务输出测试通过：纯空格/换行保留，孙任务镜像 DONE 不会提前结束当前子任务。
- 助手历史加载从接近行数上限的主 store 拆到 `assistant-history.ts`，主 store 降到约 880 行。新增识别 HTTP 200 的 CommonResult 错误，明确被服务端拒绝的请求不会一直显示运行中；网络不确定时仍只查询/接续原任务。
- 最终前端 TypeScript、相关 ESLint、24 条流式/恢复/画布结果回归通过。
- 后端现有单测报告合并全量及定向复测共 752 项，0 失败、0 错误、1 项媒体测试按条件跳过；本需求真实数据库/HTTP/SDK 集成共 18 项全部通过。统计口径为全量执行后合并定向复测。
- Maven 实际依赖树确认 Harness、Core、MySQL、OpenAI、Anthropic、Gemini、DashScope、Ollama 全部为 2.0.4。
- 未修改工具/子 agent 的业务定义和注册方式；没有引入沙盒、命令执行或文件操作工具，没有数据库迁移；本地原有未提交改动保留，未创建提交或部署现有服务。

下一步：无待完成实现。归档本需求；用户重启既有前后端服务后加载工作区新版本。

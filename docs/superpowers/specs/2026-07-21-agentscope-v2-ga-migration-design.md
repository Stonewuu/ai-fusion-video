# AgentScope Java 2.0.0 GA 迁移设计

日期：2026-07-21

状态：已获用户批准，待书面规格复核

范围：P-1（P-1A 依赖与编译契约、P-1B Runtime/Harness 收敛）

## 1. 决策摘要

本项目在当前 `main` 上分阶段将 AgentScope Java `1.0.12` 完整迁移到 `2.0.0` GA，不合并 `origin/feat-agentscope-v2`，也不长期保留 V1/V2 双运行时。

参考 `D:\develop\my\ai-work-studio` 的通用 GA 接法，但只复用以下基础设施模式：

- Harness 工厂、带界限的缓存与引用计数 lease；
- 每次调用显式创建 `RuntimeContext`；
- `AgentStateStore` 环境化工厂；
- `AgentEvent` 全枚举映射；
- `ToolBase`、`ToolCallParam` 和强类型工具上下文；
- `ChatModelBase#doStream`；
- complete/error/cancel 三终态资源释放、终态 CAS 与调用级幂等用量结算。

不复制参考项目的组织、权限、工作区、MCP、通知、定时任务、Agent 目录或其他工作台业务设计。

本规格优先保证产品健壮性、可恢复性、可审计性和多实例一致性，不以字段最少或修改最少为目标。关键选择如下：

1. 生产状态使用官方 `RedisAgentStateStore`；local/test 使用 `InMemoryAgentStateStore`。
2. 产品运行状态与 AgentScope 会话状态分离：新增 `afv_agent_run`、`afv_agent_event` 和调用级 `afv_agent_model_call_usage`，不新增 MySQL AgentState 表。
3. 事件协议使用显式的每运行 `sequence_no`，并建立 `UNIQUE(run_id, sequence_no)`；不把数据库全局自增 ID 暴露为产品序号。
4. 数据库事件日志是回放真相，Redis 只承担实时通知和补偿发布。
5. 同一 conversation 同时只允许一个活动 run；数据库生成列和唯一索引建立不可绕过的活动槽约束。
6. 现有平台子 Agent 继续作为业务工具；AgentScope 原生 `taskId` 不替代平台 `childRunId`，P-1 默认不启用原生 subagent。
7. Web/Reactor 链路直接消费 `Flux<AgentEvent>`，不调用 `.block()`、`.toIterable()`、`Thread.sleep()`，也不使用 ThreadLocal。
8. `REQUIRE_USER_CONFIRM` 和 `REQUIRE_EXTERNAL_EXECUTION` 采用可恢复等待状态，不只记录日志；确认/外部结果可以在另一节点恢复同一 StateStore session。

## 2. 范围与非目标

### 2.1 本阶段包含

- AgentScope 依赖统一为 `2.0.0` GA；
- V2 Harness、RuntimeContext、AgentStateStore 和强类型消息；
- 五类官方模型扩展及必要的自定义模型适配；
- V2 全量 AgentEvent 映射、持久事件日志、SSE 兼容与精确重连；
- ToolBase 工具适配、阻塞边界和取消传播；
- 平台子 Agent 的父子运行、事件和取消关系；
- 模型/Agent/工具/数据库/Redis 的三终态收敛；
- 后端、前端、Provider 和 Ark 的验证体系；
- 删除 V1 Hook、Session、Msg 和无效依赖。

### 2.2 本阶段不包含

- P0 的分镜清空、模型发现提示、百炼连接检测产品重构和首尾帧业务 Bug 修复；
- P1 的完整 Provider 配置页和媒体能力 schema 重构；
- P2/P3/P4 的生成工作台、资产重构和分镜组；
- 参考项目的工作区、权限、MCP、计划模式、通知和定时任务；
- 自动重放具有外部副作用的中断工具；
- 原生 subagent 产品化。

P-1 会修复阻止本阶段全量测试通过的既有测试夹具问题，但不借机实现上述后续业务需求。

## 3. 现状证据与根因

### 3.1 当前仍是完整 V1

- `ai-fusion-video/pom.xml` 依赖 `agentscope-spring-boot-starter:1.0.12` 和 `agentscope-extensions-session-mysql:1.0.12`。
- POM 仍固定 Jackson `2.17.3`，并覆盖 `json-schema-validator:3.0.0`。
- `AgentScopeAssistantService` 每次调用创建 V1 `ReActAgent`，输入被降级为旧 `Msg` 纯文本。
- `MysqlSession(dataSource, true)` 只在启动时初始化，没有实际 save/load wiring，还绕过 Flyway 执行运行时 DDL。
- `StreamingEventHook` 只覆盖少数 V1 Hook 事件，未知事件会静默遗漏。
- `AgentScopeToolAdapter` 仍实现 V1 `AgentTool`，同步工具被放入没有调度切换的 `Mono.fromCallable`。
- 自定义模型仍存在 V1 `Model`、同步迭代 Provider 流和不完整的取消边界。

### 3.2 当前运行与会话概念混用

- `conversationId` 可被多轮请求复用；conversation 是业务会话，不是一次执行。
- 当前活动 Agent、Disposable、Redis key、取消标记和回放均以 `conversationId` 为键。
- 同会话再次运行会覆盖旧运行句柄或清理旧取消状态。
- cancel 和 `doFinally` 都会无条件写 conversation 状态，没有数据库 CAS，存在完成覆盖取消、重复终态和重复结算风险。
- 通用 `TaskStreamService` 也复用 conversation/message 表，不能直接把 conversation 状态改造成 Agent run 真相。

因此需要独立 run 聚合，而不是继续向 conversation 上叠加更多运行字段。

### 3.3 当前回放不能证明不重不漏

- Redis Stream 只保留 200 条，TTL 仅 1 小时。
- Stream、Replay List、消息落库和状态更新不是原子操作。
- 重连接口不接受游标，前端只解析 `data:`，忽略 SSE `id:`。
- 任务终态后重连会直接返回空流，终态不能稳定重放。
- 连续 token 事件可能已经发送但尚未进入合并 Replay，断线后无法还原。

### 3.4 实验分支不可作为迁移基线

`origin/feat-agentscope-v2` 使用 `2.0.0-RC1`，仍保留旧 starter/session 和 schema 补丁，实际回退 deprecated `stream()`，并混入大量 UI、工作区和业务改造。只允许借鉴少量 builder 和测试思路，不合并或整体 cherry-pick。

### 3.5 当前构建基线

- Java/Javac 已恢复为 OpenJDK `21.0.2`。
- Maven Wrapper `3.9.12` 可正常启动。
- 仓库声明 `pnpm@10.32.1`，必须使用 `corepack pnpm`，不能使用 PATH 中的裸 pnpm 11。
- 迁移前全量后端测试共 111 个：94 通过、7 失败、10 错误。
- 失败集中在 `GenerationModelCapabilityServiceTests` 和 `GetGenerationModelCapabilitiesToolExecutorTests`：生产依赖从旧服务改为 `AiModelMetadataResolver` 后，测试仍传 `null`。这是既有测试夹具缺陷，不是 AgentScope 迁移结果。
- 当前 AgentScope、AiStreamRedis 和 Provider 相关的 14 个测试通过，但覆盖面不足以证明 V2。

## 4. 方案比较

### 4.1 方案 A：一次性原地大替换

一次同时修改依赖、Runtime、模型、事件、工具、SSE 和前端。优点是没有过渡代码；缺点是回归面过大，无法确定故障发生在哪一层，不采用。

### 4.2 方案 B：合并 RC1 实验分支

优点是已有部分 V2 builder 代码；缺点是依赖版本错误、API 已过期、业务改动污染严重，不采用。

### 4.3 方案 C：当前 main 上纵向可验证迁移

先建立 GA 编译契约，再跑通最小无工具 Agent，随后依次迁移事件、模型、工具、子 Agent 和取消。每个切片有独立测试和回滚点。迁移期间允许短期适配层，但最终删除全部 V1 运行时。

采用方案 C。

## 5. 目标架构

```text
Pipeline API
  -> AgentRunCoordinator
      -> RunExecutionSupervisor (owns execution subscription)
      -> RunRepository / RunTerminalCoordinator
      -> HarnessLeaseCache
          -> HarnessAgent.streamEvents(messages, runtimeContext)
              -> official/custom ChatModelBase
              -> ToolBase / platform sub-agent tools
      -> AgentEventMapper
      -> AgentEventJournal (MySQL source of truth)
      -> committed-event publisher
          -> local SSE
          -> Redis live notification / retry outbox
      -> frontend normalization + (runId, sequence) dedup

RedisAgentStateStore <-> HarnessAgent
CancellationCoordinator -> DB cancel state + Redis broadcast
                        -> delegate.interrupt(runtimeContext)
                        -> Provider dispose + tool cancel
```

### 5.1 核心组件边界

#### AgentScopeHarnessFactory

根据稳定的 Agent 定义、模型配置、Prompt、工具集合和状态存储构造 Harness。它不持有用户会话数据。

#### HarnessLeaseCache

缓存键包含：

- Agent definition stable key；
- 模型配置版本/指纹；
- Prompt 版本；
- 工具白名单版本。

缓存具备最大容量、TTL、引用计数、配置失效和关闭生命周期。`maximumSize=64` 是包括 active entry 在内的硬上限：如果 64 个不同 key 全部持有 lease，第 65 个新 key 最多等待可配置的 `5s`，仍无空槽则返回 `HARNESS_CAPACITY_EXHAUSTED`/503；绝不突破上限，也不淘汰或关闭 active entry。过期项只有在最后一个 lease 释放后才关闭；应用关闭时统一回收。

lease 获取、执行和 complete/error/cancel 清理使用单一 `Flux.usingWhen` 资源作用域，覆盖“获取后订阅前取消”与异常路径，并以幂等 close 防止双重释放。应用停机先停止新 lease、等待 drain 期限、持久中断剩余 run，再强制回收无活动 lease 的资源。

缓存项是明确的 `AgentKernelResource` 所有权聚合：它关闭自身创建的模型 client、transport、toolkit 和其他可关闭资源，不能假设 `HarnessAgent.close()` 会代劳，因为 GA delegate close 可能是 no-op。生产 `RedisAgentStateStore` 是 Spring 管理的共享单例，不属于任何 Harness cache entry；Harness eviction 不关闭它。应用停机时单独关闭 Store，`SpringStringRedisClientAdapter.close()` 对 Spring 共享连接为 no-op，避免误关全局 Redis 连接。

#### AgentScopeRuntimeContextFactory

每次运行创建新的 `RuntimeContext`，注入 user/session/conversation/run/project/pipeline/tool/cancel 上下文。所有可空项先判断再放入 builder，避免 GA `RuntimeContext.Builder.put(String, null)` 的 NPE。

#### AgentRunCoordinator

负责启动事务和活动 run 唯一性，并把运行交给 RunExecutionSupervisor。Web 层获得的是独立观察流，不拥有业务执行订阅。

#### RunExecutionSupervisor

由服务端独立持有 AgentEvent subscription、Harness lease、Provider/tool 句柄、owner lease 和终态责任。HTTP/SSE 客户端断开只卸载该观察者，不 dispose Agent；run 继续执行并持久化，随后可以 reconnect。只有显式 cancel、deadline、owner fencing、进程关闭编排或终态才停止业务执行。

应用优雅停机先停止接收新 run，给活动 run 一个可配置 drain 窗口；仍未结束的 run 通过持久取消/终态路径收敛后再释放资源。非优雅崩溃由 owner lease 对账处理，不能只依赖 JVM shutdown hook。

AgentScope 内部使用 `Flux.create(..., BUFFER)`，不能依赖下游背压自动限制。Supervisor 以快速 ingress subscriber 请求上游并同步写入有界 MPSC 队列，journal worker 再串行消费；队列同时限制事件数和估算字节数。默认 `4096` 事件或 `8MiB`，任一超限立即取消上游、关闭 Provider、记录指标并把 run 收敛为 `AGENT_EVENT_BACKPRESSURE_OVERFLOW`，不允许无界堆积导致 JVM OOM。

#### AgentEventJournal

对每个 run 串行分配 sequence、持久化事件并在事务提交后发布。数据库是回放真相；事件表同时承担可靠发布 outbox。

#### RunTerminalCoordinator

完成、失败、取消都经过同一入口。它在一个事务内完成 run 状态条件更新、终态事件写入和活动槽释放；只有条件更新影响一行时，终态才获胜。

#### CancellationCoordinator

接收 runId，持久化取消请求，广播到所有节点，命中实际 owner 后定向中断 Agent、Provider 和可取消工具。数据库状态用于补偿丢失的 Redis 通知。

#### AgentRuntimeSchedulers 与 StateStore fail-closed

AgentScope GA 的 `AgentStateStore` 是同步接口，当前 MyBatis/JDBC 事件日志也是阻塞 I/O。为保证 Web/Reactor event-loop 不被占用，建立三个有界、可观测、可拒绝过载的调度边界：

- `agent-state`：Harness 激活、RedisAgentStateStore load/save 和 session 清理；
- `agent-journal`：run/event/message 的 MyBatis 事务；
- `agent-model-blocking`：同步 Provider 的连接建立和流读取；
- `agent-tool-blocking`：暂时无法改成真正异步的白名单工具。

Web 入口以 `Flux.defer(...)` 延迟创建 Harness 事件流，并在 `agent-state` Scheduler 上完成调用前 Redis 可用性/状态预检；事件持久化使用 `concatMap` 切到 `agent-journal`。不得用无限队列或无界线程池掩盖阻塞，队列满时返回明确过载错误并记录指标。

GA Core 在状态 load 异常时会捕获异常并创建 fresh state，save 又固定调度到 Reactor 全局 `boundedElastic`。因此不能声称外层 `subscribeOn` 可以控制所有内部 load/save。采用以下不修改官方 Core 的 fail-closed 方案：

- `FailClosedAgentStateStore`/Redis adapter 装饰器按 `(userId, sessionId)` 记录任何 load/save/exists/delete 失败并继续向上抛出；
- 每次 run 开始前执行真实 Redis 预检并清除旧失败标记；
- 所有模型通过 `StateStoreGuardedChatModel` 装饰器、所有平台工具通过 `AbstractPlatformAgentTool` 基类，在调用前检查 StateStore guard；若 GA 已吞掉 load 异常，guard 从失败标记发现后在调用 Provider/工具前终止 run，禁止以空状态继续；
- terminal coordinator 在 COMPLETED 前再次检查 save 失败标记；保存失败时只能收敛为 FAILED；
- adapter 设置命令超时、信号量 bulkhead、并发/队列指标，限制 GA 内部 `boundedElastic` 上的同步 Redis 调用；不 fork AgentScope Core，也不假装把它改到了自定义 Scheduler。

### 5.2 可配置运行默认值

首版提供显式配置并使用以下保守默认值，避免把关键容量藏在代码常量中：

- Harness cache：`maximumSize=64`、`expireAfterAccess=30m`；
- AgentEvent ingress：最多 `4096` 个待处理事件或 `8MiB` 估算 payload；
- 文本/思考事件 chunk：`maxDelay=50ms`、`maxChars=1024`，遇到 block end、工具、错误、取消或终态立即 flush；
- owner heartbeat：每 `5s`；lease `20s`；对账扫描每 `5s`，均使用数据库时间判断；
- outbox 重试：从 `100ms` 指数退避到 `30s`，带 jitter，持续重试并在积压超阈值时告警；
- `agent-state` 和 `agent-journal` 线程数默认 `max(4, min(32, CPU*2))`，队列分别为 512/2048；
- `agent-model-blocking` 线程数默认 `max(8, min(64, CPU*4))`、队列 256，并受 Provider/模型自身并发许可进一步限制；
- `agent-tool-blocking` 线程数默认 `max(8, min(64, CPU*4))`、队列 256；工具自身的并发许可仍可更低；
- cancel flag 的 TTL 至少为该 run 绝对 deadline 加 1 小时，终态持久化后再延迟清理。
- 用户确认默认等待 `30m`；外部执行等待不超过 run 的绝对 deadline；等待期间释放 owner lease 和 Provider 资源。

测试环境可以缩短时间参数，但生产默认值只能通过配置覆盖，不在测试代码中改变全局行为。所有 Scheduler、cache、lease 和 outbox 参数暴露指标。

## 6. AgentScope 2.0.0 GA 契约

### 6.1 依赖

所有 AgentScope 依赖使用统一属性 `${agentscope.version}=2.0.0`：

- `io.agentscope:agentscope-harness`
- `io.agentscope:agentscope-extensions-redis`
- `io.agentscope:agentscope-extensions-model-openai`
- `io.agentscope:agentscope-extensions-model-anthropic`
- `io.agentscope:agentscope-extensions-model-gemini`
- `io.agentscope:agentscope-extensions-model-dashscope`
- `io.agentscope:agentscope-extensions-model-ollama`

`agentscope-core` 由 Harness 以 compile scope 传递；编译契约测试可以显式引用其 API，但不再引入旧聚合 starter。

删除：

- `agentscope-spring-boot-starter`
- `agentscope-extensions-session-mysql`
- V1 `MysqlSession`
- V1 Hook/Msg/Model/Session 兼容代码

### 6.2 精确 Runtime API

生产调用使用：

```java
harnessAgent.streamEvents(messages, runtimeContext)
```

定向取消使用：

```java
harnessAgent.getDelegate().interrupt(runtimeContext)
```

不能调用 Harness 无参 `interrupt()`，因为它走 deprecated 默认会话路径。

工具从 `ToolCallParam#getRuntimeContext()` 获取类型化上下文；`getContext()` 已 deprecated。

### 6.3 AgentStateStore

环境策略：

- local：应用级共享 `InMemoryAgentStateStore` Bean，同一进程内跨 Harness 淘汰仍能恢复，但进程重启不承诺持久化；
- test：每个测试 ApplicationContext 显式创建并在结束时清理独立 `InMemoryAgentStateStore`，避免测试间串状态；
- production：`RedisAgentStateStore.builder().clientAdapter(...).keyPrefix("afv:agentscope:v2:").build()`。

复用参考项目的 `SpringStringRedisClientAdapter` 思路，让官方 Store 使用现有 Spring Redis 连接配置和生命周期管理；不假设 `RedisConnectionFactory` 能直接传给官方 builder。

官方 Redis Store 没有覆盖单 key `delete(userId, sessionId, key)`，该方法实际为默认 no-op。因此清理以完整 `(userId, sessionId)` 为单位；不得设计依赖逐 state key 删除的逻辑。

AgentStateStore 只保存 AgentScope Runtime 状态，不保存平台 run/event 真相。P-1 不复制 Redis+MySQL 双层 tombstone/read-repair 实现，也不引入 DistributedStore、workspace 或 sandbox。

由于官方 Store API 同步，应用主动执行的预检和 session 清理在 `agent-state` 有界 Scheduler 上运行。GA 内部 save 使用其固定的 `boundedElastic`；通过 adapter bulkhead/timeout/失败标记控制，而不是依赖外层调度。不能仅因为调用发生在 AgentScope 内部就假设它是非阻塞操作。

### 6.4 依赖兼容

GA Core 的依赖基线与当前补丁不同。实施时必须先生成并保存以下依赖树证据：

- Jackson 全部模块及版本；
- Reactor；
- `json-schema-validator`；
- Ark SDK；
- AgentScope 所有模块；
- Lettuce/Jedis/Redisson 与 Spring Redis 的实际选择。

目标是删除 Jackson `2.17.3` 降级 BOM 和 `json-schema-validator:3.0.0` 覆盖，但只有在依赖树、编译、测试和 Ark smoke 均通过后才能完成。火山 Ark SDK 升级到已核验修复旧 Jackson 命名 API 的 `2.0.19`，再移除降级 BOM。

Redis 扩展 Builder 的公开签名直接引用 Jedis、Lettuce 和 Redisson 类型，粗暴 exclusion 未使用客户端可能在类加载/反射时触发 `NoClassDefFoundError`。P-1 先保留扩展声明的三类客户端依赖并用依赖树验证无版本冲突；只有通过单独的类加载与 Spring 启动测试证明安全后，才允许精确 exclusion，不能以减包体积为由冒运行时风险。

## 7. Harness、上下文与状态隔离

### 7.1 稳定寻址

AgentScope Store 按 `(userId, sessionId)` 寻址：

```text
userId    = 当前认证用户的稳定字符串 ID
sessionId = afv:v2:{conversationId}:{agentDefinitionStableKey}
```

平台子 Agent 使用独立的 `agentDefinitionStableKey`，避免与主 Agent 共享 state slot。不得把临时 runId 用作需要跨轮恢复的主会话 sessionId。

### 7.2 类型化业务上下文

通过 `put(Class<T>, T)` 注入以下不可变对象：

- `AuthenticatedUserContext`
- `AgentConversationContext`
- `AgentRunContext`
- `ProjectContext`
- `PipelineRequestContext`
- `ToolExecutionContext`
- `CancellationContext`

工具不得通过构造器捕获一次请求的可变上下文，也不得使用 ThreadLocal。

### 7.3 并发与恢复

- Harness 可以跨用户复用；RuntimeContext 和 StateStore slot 必须隔离。
- 同一 `(userId, sessionId)` 的运行由 conversation 活动槽串行化。
- 每次调用开始时从生产 Store 重新加载状态，避免请求漂移到其他节点后读取本机陈旧状态。
- 重建 Harness 后使用相同 `(userId, sessionId)` 恢复。
- conversation 删除或满足明确的数据保留策略时，删除完整 session。

## 8. 强类型消息与媒体

### 8.1 消息

使用 `UserMessage`、`AssistantMessage` 和 V2 ContentBlock：

- 文本保持文本块；
- 图片/视频保持媒体块；
- 工具调用保持 ToolUse；
- 工具结果保持 ToolResult；
- 不把多模态或工具内容拼成普通字符串。

`UserMessage`/`AssistantMessage` 的 role 固定，不调用会抛 `UnsupportedOperationException` 的 builder `role(...)`。

### 8.2 媒体

至少支持：

- `new URLSource(url, mimeType)`；
- `new Base64Source(mediaType, data)`；
- 正确 MIME；
- 保留内容块和引用顺序；
- 为视频等后续块类型保留扩展点。

Agent 层只表达输入，实际 URL/Base64 选择继续复用现有模型能力解析。媒体 resolver 必须：

- 拒绝任意 `file://` 路径和越权本地文件；
- 对远程下载执行 SSRF、重定向、私网地址、超时、MIME 和大小限制；
- 不把 Base64 写入日志或平台消息表；
- URL-only 模型在无法获得 Provider 可访问的绝对 URL 时入模前失败；
- 不通过失败重试猜测 Provider 传输协议。

## 9. AgentEvent 映射

### 9.1 全量枚举

GA `AgentEventType` 共 31 项，映射器必须以穷举 switch 覆盖：

```text
AGENT_START, AGENT_END, AGENT_RESULT,
MODEL_CALL_START, MODEL_CALL_END,
TEXT_BLOCK_START, TEXT_BLOCK_DELTA, TEXT_BLOCK_END,
THINKING_BLOCK_START, THINKING_BLOCK_DELTA, THINKING_BLOCK_END,
DATA_BLOCK_START, DATA_BLOCK_DELTA, DATA_BLOCK_END,
TOOL_CALL_START, TOOL_CALL_DELTA, TOOL_CALL_END,
TOOL_RESULT_START, TOOL_RESULT_TEXT_DELTA,
TOOL_RESULT_DATA_DELTA, TOOL_RESULT_END,
EXCEED_MAX_ITERS,
REQUIRE_USER_CONFIRM, REQUIRE_EXTERNAL_EXECUTION,
USER_CONFIRM_RESULT, EXTERNAL_EXECUTION_RESULT, REQUEST_STOP,
SUBAGENT_EXPOSED, HINT_BLOCK, ALL_TOOLS_DENIED, CUSTOM
```

每个事件保留：

- `source`
- `replyId`
- `blockId`
- `toolCallId`
- `rawEventId`
- `rawEventType`
- `createdAt`
- `runId`
- `sequence`

这些 identity 由具体 subtype 分支提取；某类事件原本没有 replyId/blockId/toolCallId 时 envelope 对应字段为 NULL，不能调用不存在的统一 getter 或伪造值。

新增上游枚举必须导致编译或完整性测试失败；`CUSTOM` 也必须持久化并显式记录，不能用 default 静默丢弃。

事件进入 Journal 前经过 `AgentEventEnvelopeSanitizer`：保留结构和 identity，但剔除 API key、代理凭据、Authorization、原始 Base64/二进制、越权文件路径和不应持久化的签名参数。媒体使用受控 asset/media 引用、MIME、长度和内容哈希表示。这里的“完整 raw payload”指完整事件语义，不意味着持久化秘密或大块二进制。

### 9.2 双层投影

映射器输出版本化 `AgentEventEnvelope`，其中包含：

1. 完整 raw identity/payload；
2. 可选 legacy SSE projection。

非渲染生命周期事件仍写数据库和指标，但不伪装成 `CONTENT`。旧 UI 只接收已有 outputType；未来 UI 可通过 schemaVersion 使用更丰富的控制事件。

### 9.3 旧 SSE 类型

保持：

- `REASONING`
- `CONTENT`
- `TOOL_CALL`
- `TOOL_FINISHED`
- `SUB_AGENT_FINISHED`
- `DONE`
- `ERROR`
- `CANCELLED`

主 Agent 的 `DONE/ERROR/CANCELLED` 必须继续保持：

```text
parentToolCallId = null
agentName = null
```

否则现有通知面板无法 settle。

### 9.4 用户确认与外部执行

`REQUIRE_USER_CONFIRM` 不是终态，但收到该事件时也不能立刻 dispose：GA 随后还会产生 stop/result 并在 call 生命周期末保存 AgentState。正确时序是：

1. 先把 `REQUIRE_USER_CONFIRM` 作为 raw-only pending candidate 持久化，不向前端发布可操作按钮；
2. 保持当前 stream 自然运行到匹配 replyId 的暂停 `AgentResult`/正常结束，不主动取消订阅；
3. 确认 StateStore save 没有失败标记后，在一个事务内把 run 从 RUNNING CAS 为 WAITING_CONFIRMATION，记录 `waiting_reply_id/wait_expires_at`，清理 owner instance/lease 但保留 epoch 历史，并写平台合成 `PLATFORM_USER_CONFIRM_REQUIRED` 兼容事件；
4. 事务提交后释放 Harness lease；RedisAgentStateStore session 保留；只有此时才通过 `TOOL_CALL` 和可选 `controlType=USER_CONFIRM_REQUIRED`、`pendingToolCalls`、`expiresAt` 向前端展示确认操作；
5. 若进程在 candidate 与状态保存确认之间崩溃，不发布可操作确认事件；lease 对账将 run 收敛为 OWNER_LOST，而不是让用户恢复一个未持久化状态；
6. 用户提交的请求只包含 replyId、toolCallId 和批准/拒绝决定，服务端从已持久化 pending event 重建原始 ToolUseBlock，不信任客户端回传工具名、input 或 schema；
7. 确认事务把 WAITING_CONFIRMATION CAS 回 RUNNING，领取节点写入自己的 owner instance、递增 owner epoch、建立新 lease，并写 USER_CONFIRM_RESULT 事件；提交后用相同 userId/sessionId 和带 `Msg.METADATA_CONFIRM_RESULTS` 的 UserMessage 重新调用 `streamEvents`，因此恢复执行可以由另一节点领取。

当前 GA 外部工具路径不保证直接发出 `REQUIRE_EXTERNAL_EXECUTION/EXTERNAL_EXECUTION_RESULT`：外部工具会返回 suspended ToolResult，最终 `AgentResult.Msg.generateReason=TOOL_SUSPENDED`。因此 P-1 从已成功保存状态的暂停 AgentResult 合成 `PLATFORM_REQUIRE_EXTERNAL_EXECUTION` 并进入 WAITING_EXTERNAL；上游枚举事件若未来出现仍由全枚举 mapper 保存。外部结果只能由受认证的内部执行器提交，服务端按 replyId/toolCallId 校验 pending event 后构造匹配的 `ToolResultBlock` 恢复消息，并写平台合成 result audit。P-1 不开放任意用户提交外部工具结果。

确认超时、外部执行超时、取消和重复响应均通过数据库状态 CAS 收敛。WAITING 状态没有活动 Provider；取消时可以直接进入 CANCEL_REQUESTED/CANCELLED，不等待本机 Disposable。

## 10. 数据模型与增量 Migration

仓库当前最新 Migration 为 `V1.0.6.1.4__storyboard_episode_script_link.sql`。实施前先查询目标库 `flyway_schema_history`；若没有并行版本冲突，新增：

```text
V1.0.6.1.5__agent_run_and_event.sql
```

不得修改 `V1__init_mysql.sql` 或任何已执行 Migration。

### 10.1 `afv_agent_run`

采用以下字段：

| 字段 | 约束 | 用途 |
|---|---|---|
| `id` | bigint PK auto_increment | 内部主键 |
| `run_id` | varchar(64) not null unique | 对外稳定运行 ID，复用当前每次调用生成的 messageId |
| `conversation_id` | varchar(64) not null | 关联现有多轮会话 |
| `user_id` | bigint not null | 授权与审计快照 |
| `project_id` | bigint null | 项目上下文快照 |
| `agent_type` | varchar(64) null | Agent 定义审计 |
| `kernel_fingerprint` | varchar(64) not null | 本次 Harness 构建快照哈希/缓存键 |
| `agent_definition_snapshot_json` | mediumtext not null | 脱敏的 Agent、模型、Prompt、工具 manifest 快照 |
| `status` | varchar(32) not null | RUNNING/WAITING_CONFIRMATION/WAITING_EXTERNAL/CANCEL_REQUESTED/COMPLETED/FAILED/CANCELLED |
| `owner_instance_id` | varchar(128) null | 实际运行节点 |
| `owner_epoch` | bigint not null default 0 | owner fencing token；每次领取递增 |
| `lease_until` | datetime(3) null | 节点失联检测 |
| `next_sequence` | bigint not null default 1 | 数据库串行分配下一事件序号 |
| `terminal_sequence` | bigint null | 唯一终态序号 |
| `terminal_output_type` | varchar(32) null | DONE/ERROR/CANCELLED |
| `cancel_requested_at` | datetime(3) null | 取消审计 |
| `cancel_broadcast_at` | datetime(3) null | 最近广播时间 |
| `cancel_acknowledged_at` | datetime(3) null | owner 已观察取消 |
| `cancel_next_attempt_at` | datetime(3) null | 丢通知补偿扫描 |
| `waiting_reply_id` | varchar(128) null | 当前等待确认/外部执行的 reply |
| `wait_expires_at` | datetime(3) null | 等待超时 |
| `started_at` | datetime(3) not null | 开始时间 |
| `heartbeat_at` | datetime(3) null | owner 心跳 |
| `finished_at` | datetime(3) null | 结束时间 |
| `error_code` | varchar(64) null | 机器可读错误 |
| `error_message` | text null | 脱敏错误摘要 |
| `usage_settled` | tinyint not null default 0 | 全部调用用量已幂等结算 |
| `usage_settled_at` | datetime(3) null | 结算时间 |
| `projected_through_sequence` | bigint not null default 0 | 消息投影已处理游标 |
| `projection_completed_at` | datetime(3) null | 终态已完整投影 |
| `create_time/update_time` | datetime(3) | 审计时间 |

关键约束与索引：

- `UNIQUE(run_id)`；
- 增加生成列 `active_conversation_id`：状态为 RUNNING/WAITING_CONFIRMATION/WAITING_EXTERNAL/CANCEL_REQUESTED 时取 `conversation_id`，终态时为 NULL；
- `UNIQUE(active_conversation_id)`，利用 MySQL 允许多个 NULL，保证每个 conversation 最多一个活动 run，避免应用漏写手工 active flag；
- `CHECK` 限制 status 枚举；部署前校验 MySQL `>=8.0.16`，低于该版本拒绝执行 Migration，不能假设 CHECK 会生效；
- `(conversation_id, status, id)`；
- `(user_id, status, update_time)`；
- `(status, lease_until)` 用于失联 run 对账。

snapshot 不保存 API key、代理密码或其他秘密，只保存不可变配置 ID/版本、模型参数、Prompt hash/内容、工具名/schema/readOnly/concurrencySafe manifest 及代码版本。WAITING 状态恢复时必须用该 snapshot/fingerprint 重建原 Kernel；如果原模型/Agent 版本或兼容工具实现已经不可获得，则 fail closed 为 `RUN_CONFIG_UNAVAILABLE`，不能拿最新配置执行旧 pending ToolUse。

### 10.2 `afv_agent_event`

采用以下字段：

| 字段 | 约束 | 用途 |
|---|---|---|
| `id` | bigint PK auto_increment | 内部存储 ID，不作为产品序号 |
| `run_id` | varchar(64) not null | 所属运行 |
| `sequence_no` | bigint not null | 每运行严格递增序号 |
| `schema_version` | int not null default 1 | 事件 envelope 版本 |
| `raw_event_id` | varchar(128) null | AgentScope 原始 ID |
| `raw_event_type` | varchar(64) not null | 31 类原始事件或 PLATFORM_* 合成事件 |
| `source` | varchar(255) null | main/subagent/tool 来源 |
| `reply_id` | varchar(128) null | 原始 reply identity |
| `block_id` | varchar(128) null | 原始 block identity |
| `tool_call_id` | varchar(128) null | 工具调用 identity |
| `parent_tool_call_id` | varchar(128) null | 平台父调用关系 |
| `agent_name` | varchar(128) null | 兼容投影字段 |
| `output_type` | varchar(32) null | legacy SSE 类型；非渲染事件为空 |
| `payload_json` | mediumtext not null | 完整版本化 envelope，含可选 legacy projection |
| `event_created_at` | datetime(3) null | 上游事件时间 |
| `redis_published_at` | datetime(3) null | 可靠发布 outbox 状态 |
| `publish_required` | tinyint not null | raw-only 为 0，legacy SSE 为 1 |
| `publish_status` | varchar(16) not null | NOT_REQUIRED/PENDING/CLAIMED/PUBLISHED |
| `publish_claim_owner` | varchar(128) null | 多实例 claim owner |
| `publish_claim_until` | datetime(3) null | claim lease |
| `next_publish_attempt_at` | datetime(3) null | 退避重试时间 |
| `last_publish_error` | varchar(1024) null | 脱敏发布错误 |
| `publish_attempts` | int not null default 0 | 发布补偿计数 |
| `create_time` | datetime(3) not null | 持久化时间 |

关键约束与索引：

- `UNIQUE(run_id, sequence_no)`；
- `(run_id, output_type, sequence_no)`；
- `(publish_status, next_publish_attempt_at, id)` 用于扫描未发布事件；
- `(run_id, raw_event_id)` 普通索引用于审计，不假设上游 raw ID 全局唯一。

### 10.3 `afv_agent_message` 和 conversation 顺序

增量增加：

- `afv_agent_message.run_id varchar(64) null`；旧历史保持 NULL；
- `afv_agent_message.projection_key varchar(64) null`，使用稳定 SHA-256 十六进制幂等键；
- `UNIQUE(conversation_id, message_order)`；
- `UNIQUE(projection_key)`，NULL 允许旧消息和非投影消息共存；
- `(conversation_id, run_id, message_order)` 索引；
- `afv_agent_conversation.next_message_order bigint not null default 1`，默认值至少保留到 V1 回滚窗口关闭。

Migration 先检测历史重复 `message_order`，按 `(message_order, id)` 稳定重排，再增加唯一约束；序号计算包含逻辑删除消息，避免重用审计顺序。随后按每个 conversation 的全量 `MAX(message_order)+1` 回填 `next_message_order`，并把 `message_count` 校正为实际未删除消息数；不修改任何消息 ID、toolCallId、资产、分镜或任务 ID。

新消息通过短事务锁定 conversation 行、预留 message order，再写消息，并在同一事务更新 `message_count/last_message_time`；不再使用并发不安全的 `MAX(message_order)+1`。`AgentConversationService.createOrUpdate` 不再把“开始一次 run”误计为一条消息。通用 TaskStream 也必须复用同一 allocator，避免两套写入者再次竞争。

### 10.4 `afv_agent_model_call_usage`

一个 ReAct run 可以多次调用模型，不能用 runId 把后续调用误判为重复。新增调用级用量账本：

| 字段 | 约束 | 用途 |
|---|---|---|
| `id` | bigint PK auto_increment | 内部主键 |
| `run_id` | varchar(64) not null | 所属 run |
| `model_call_id` | varchar(64) not null | 调用前生成并持久化的稳定 ID |
| `provider` | varchar(64) not null | Provider |
| `model_code` | varchar(128) not null | 实际模型 |
| `status` | varchar(24) not null | STARTED/COMPLETED/FAILED/CANCELLED |
| `input_tokens/output_tokens/reasoning_tokens/cache_tokens` | bigint null | 标准化用量 |
| `usage_json` | mediumtext null | 脱敏的 Provider 原始用量 |
| `settlement_status` | varchar(24) not null | PENDING/CLAIMED/SETTLED |
| `settlement_attempts` | int not null default 0 | 至少一次重试 |
| `next_settlement_attempt_at` | datetime(3) null | 退避调度 |
| `started_at/finished_at/create_time/update_time` | datetime(3) | 审计时间 |

建立 `UNIQUE(run_id, model_call_id)`。每次 Provider 请求发出前先写 STARTED；流完成/失败/取消时条件更新同一行。结算以 `(runId, modelCallId)` 作为下游幂等键，或在所有调用账本完结后聚合并以 runId 结算；无论选择哪条路径，都从持久账本计算，不能只依赖进程内 token 累加。run 的 `usage_settled` 仅表示其全部 model-call 行均已 SETTLED。

### 10.5 不新增 AgentState 表

AgentScope Runtime 状态只进入 RedisAgentStateStore。`afv_agent_run/event/model_call_usage/message` 是平台运行审计、重连、结算和历史投影，不是另一个 AgentStateStore。

## 11. 事件事务、发布与回放

### 11.1 启动事务

在一个短事务中：

1. 校验 conversation 归属；
2. 创建或更新 conversation；
3. 插入 RUNNING run，并写当前 `owner_instance_id`、初始 `owner_epoch=1` 和基于数据库时间的 lease；生成列自动占用该 conversation 的活动槽；
4. 提交后才获取 Harness lease 并启动 Provider。

唯一活动槽冲突转换为明确的 409，不随机选择或覆盖旧运行。

启动事务同时通过统一 message allocator 分配并写入带 runId 的用户消息；不能先提交 run、再用另一个不可靠步骤保存用户输入。

事务提交后若 Harness 获取、StateStore 激活或 Provider 启动失败，必须通过正常 terminal coordinator 将该 run 收敛为 FAILED；不能遗留永久 RUNNING 行。

### 11.2 普通事件事务

每个 run 由单一 journal writer 串行消费事件。写入时锁定 run 行，先确认状态仍为 RUNNING，再读取并递增 `next_sequence`、插入 event，然后提交。只有提交后的事件才能进入 SSE/Redis。

一旦状态进入 CANCEL_REQUESTED 或任一终态，普通内容、工具结果和成功事件不再入 Journal；取消诊断与 CANCELLED 终态只通过 Cancellation/Terminal Coordinator 写入。这样终态事务获胜后，迟到的 Provider 或工具信号不会出现在终态之后。

Reactor 实现使用保持顺序的 `concatMap`，数据库事务在 `agent-journal` Scheduler 执行；不在 Netty event-loop 直接调用 MyBatis。

为避免逐 token 数据库写放大，文本和思考 delta 在非常短的时间/大小窗口内合并成 committed chunk；未经提交的 chunk 不向客户端发送。这样断线恢复和实时显示使用同一事实，不出现“看过但无法重放”的 token。

### 11.3 可靠发布

event 表本身是 outbox：

- 本机连接收到提交后的事件；
- publisher 先用短事务 CAS 把到期 PENDING 行 claim 为 CLAIMED 并设置 claim lease，然后在事务外调用 Redis；
- Redis 发布成功后以 claim owner 条件更新为 PUBLISHED 并填写 `redis_published_at`；
- 失败时记录脱敏错误、增加 attempts、计算带 jitter 的 `next_publish_attempt_at` 并回到 PENDING；claim 过期可由其他实例接管；
- Redis 重复投递允许发生，客户端用 `(runId, sequence)` 去重；
- Redis 裁剪、重启或发布失败不影响数据库重连。

只有存在 legacy projection、即 `output_type IS NOT NULL` 的事件进入现有 Redis/SSE 通道，并写为 `publish_required=1/publish_status=PENDING`。raw-only 生命周期事件写为 `publish_required=0/publish_status=NOT_REQUIRED`，仍占用 sequence，因此前端必须允许合法 sequence gap；它们不会被补偿器扫描。旧 reconnect 查询同样只投影 `output_type IS NOT NULL` 的事件。

消息投影器按 sequence 消费 committed event，用 `projection_key` 实现幂等写入，并条件推进 `projected_through_sequence`。终态投影成功后设置 `projection_completed_at`；异常不能吞掉，由独立补偿器从游标继续。

P-1 不擅自删除历史事件。后续清理的可查询条件为：run 已终态、所有 required outbox 已 PUBLISHED、`projected_through_sequence >= terminal_sequence`、`projection_completed_at IS NOT NULL`，且超过明确的数据保留策略。若部署尚未配置保留策略，则保留事件并通过表大小/增长率告警，不用隐式 TTL 丢失审计记录。

conversation 逻辑删除、用户数据擦除、Redis session 清理和审计保留是四种不同语义：逻辑删除不自动物理删除事件；依法执行用户擦除时通过专用事务/作业按 userId 清理消息、event、run 和完整 AgentState session，并记录不含内容的擦除审计。

### 11.4 终态事务

成功/失败只允许：

```text
RUNNING -> COMPLETED | FAILED
WAITING_CONFIRMATION | WAITING_EXTERNAL -> FAILED（超时或恢复校验失败）
```

取消：

```text
RUNNING | WAITING_CONFIRMATION | WAITING_EXTERNAL
  -> CANCEL_REQUESTED -> CANCELLED
```

终态事务锁定 run，校验允许的源状态，分配 terminal sequence，插入 terminal event，更新 terminal 字段、释放 active slot。条件不满足表示另一终态已获胜，不再发布第二个终态或结算费用。

### 11.5 失联 owner

owner 只允许使用 `(runId, owner_instance_id, owner_epoch, status=RUNNING)` 条件和数据库时间更新 heartbeat/lease。影响行数为 0 时必须立即读取 run：若为 CANCEL_REQUESTED 则本机中断；若 owner/epoch 已变化或 lease 已丢失，则停止 Provider、工具和业务回填。可产生业务副作用的工具在提交前调用 RunLeaseGuard 校验相同 fencing token。

取消补偿器扫描 `CANCEL_REQUESTED` 且尚未 acknowledged/到达 next attempt 的 run 并重新广播；owner 的每次 heartbeat 也读取当前状态，因此即使 Redis 广播全部丢失，仍能观察数据库取消请求。owner 开始中断时以 owner epoch 条件写 `cancel_acknowledged_at`。

对账器处理 RUNNING 的过期 lease 时，锁行并再次用数据库时间确认过期：

- 已请求取消：收敛为 CANCELLED；
- 未请求取消：收敛为 FAILED，错误码 `OWNER_LOST`；
- 不自动重放模型或有副作用工具，避免重复生成和重复写业务数据；
- 仍通过正常 terminal journal/publish 路径通知前端。

### 11.6 Replay 到 Live 的无缝切换

Redis 只作为“数据库可能有新事件”的 wake-up，不能直接把其 payload 当成有序事实。reconnect 算法固定为：

1. 先订阅该 run 的 Redis wake-up，并建立有界提示缓冲；缓冲溢出只设置 dirty 标记，不丢数据库事实；
2. 查询数据库当前最大 committed sequence 作为 watermark `W`；
3. 按 sequence 从 `afterSequence` 回放到 `W`，只输出 legacy projection；
4. 回放期间到达的 Redis 提示只触发 tail query，不直接输出；
5. 回放完成后反复查询 `sequence > lastEmitted`，直到一次查询为空，再进入 live；
6. live 阶段每次 Redis 提示都从数据库 tail query；另有低频数据库 poll 补偿完全丢失的通知；
7. 服务端按 sequence 排序并去重后才发送，前端再以 `(runId, sequence)` 做第二层幂等；
8. 只有在 terminal sequence 已发送且一次 tail query 确认没有更晚合法事件后才关闭连接。

因此不存在“先查 DB、后订阅 Redis”造成的窗口，也不会因 Redis 乱序让前端先看到高 sequence 后永久丢弃低 sequence。

## 12. API 与前端兼容

### 12.1 保留路径

- `POST /api/ai/pipeline/run`
- `POST /api/ai/pipeline/cancel?conversationId=`
- `POST /api/ai/pipeline/confirm`
- `POST /api/ai/pipeline/external-result`（仅内部执行器）
- `GET /api/ai/pipeline/reconnect`
- `GET /api/ai/pipeline/status`
- `GET /api/ai/pipeline/running`

### 12.2 兼容扩展

- 后端流接口返回 `Flux<ServerSentEvent<AiChatStreamRespVO>>`，每条 legacy projection 设置标准 SSE `id="{runId}:{sequence}"`，不再只写 `data:`；
- run 的首个事件和后续事件携带 `runId`；
- cancel 增加可选 `runId`，旧 conversationId 解析为唯一活动 run；
- confirm 接收 `runId/replyId/decisions[{toolCallId, confirmed}]`，不接收或信任工具 input；
- external-result 使用内部认证并校验 pending reply/tool identity；
- reconnect 增加可选 `runId`、`afterSequence`，同时支持 `Last-Event-ID`；
- status/running 增加可选 `runId`、`lastSequence`；
- 所有 run/cancel/confirm/status/reconnect 必须 join conversation.user_id 做当前用户授权；confirm 还必须校验 WAITING_CONFIRMATION、replyId、未过期和 decision 集合与 pending toolCallId 完全匹配；external-result 使用独立内部执行器身份与最小权限；
- 旧客户端不传 cursor 时从该 run 的起点回放，再由前端去重；
- 已终态 run 也能回放 terminal，不返回空流。

游标优先级与校验固定为：请求中的 runId 先完成授权和目标解析；`Last-Event-ID` 若存在必须解析成相同 runId；`afterSequence` 与 header 同时存在时必须相等，否则返回 400；两者都不存在时从 0 开始。JSON 中的 runId/sequence 是客户端持久游标和第二层校验，不替代标准 SSE id。前端手写 authenticated fetch 解析并保存 `id:`，重连时显式发送 `Last-Event-ID`。

SSE VO 只增加可选字段：

- `schemaVersion`
- `runId`
- `sequence`
- `source`
- `replyId`
- `blockId`
- `rawEventId`
- `rawEventType`
- `createdAt`
- `controlType`
- `pendingToolCalls`
- `expiresAt`

### 12.3 前端改造边界

保留：

- Zustand pipeline store 的 rAF 批处理；
- 缺失父 TOOL_CALL 时的 placeholder 恢复；
- 工具成功后的资产/剧本/分镜刷新映射；
- 通知面板、历史 timeline 和专用工具结果 renderer；
- 取消、自动重连和任务恢复。

新增：

- 独立 `normalizePipelineEvent`；
- `(runId, sequence)` 去重和严格递增检查；
- Replay/live 交界去重；
- unknown schema/raw event 显式告警；
- WAITING_CONFIRMATION 工具确认操作、过期和重复提交反馈；
- `Last-Event-ID`/JSON cursor 解析；
- 终态后刷新恢复。

为 normalization 和 reducer 增加最小 Vitest 测试基础设施，不依赖人工点击证明重复、乱序和终态语义。

## 13. 模型适配

### 13.1 官方扩展优先

OpenAI、Anthropic、Gemini、DashScope 和 Ollama 优先使用官方 GA 扩展的构造和协议实现。

只保留官方实现不能覆盖的：

- OpenAI Responses；
- 代理 Transport；
- Gemini tool-response formatter；
- Vertex 定制；
- Anthropic 代理 fallback。

每个保留项必须有契约测试证明必要性；若官方扩展已经满足，删除重复自定义代码。

### 13.2 自定义 ChatModelBase

实现精确签名：

```java
protected Flux<ChatResponse> doStream(
    List<Msg> messages,
    List<ToolSchema> tools,
    GenerateOptions options)
```

要求：

- `doStream` 入口使用 `Flux.deferContextual` 读取 `AgentBase.RUNTIME_CONTEXT_KEY`，从 RuntimeContext 获得 runId、ownerEpoch、deadline 和类型化业务上下文；缺失时在 Provider 调用前 fail closed；
- 每次调用在 Provider 请求前生成稳定 modelCallId 并写入调用级 usage 账本，不能把一次 run 内的多次模型调用共用一个调用幂等键；
- Provider 连接创建也处于正确调度边界；
- 使用 `Flux.using/usingWhen` 管理 Provider stream/client；
- complete/error/cancel 均关闭；
- dispose 能中断阻塞的 Provider 读取；
- 用量收集采用至少一次重试，并以 `(runId, modelCallId)` 或持久账本聚合后的 runId 作为下游幂等键实现“业务效果一次”；全部调用确认成功后才 CAS run `usage_settled=1`；结算失败由对账任务重试，不回滚已经成立的运行终态；
- 日志不输出 API key、代理密码、完整 Base64 或敏感请求体。

## 14. 工具与子 Agent

### 14.1 ToolBase

工具统一实现 `ToolBase/AgentTool + callAsync(ToolCallParam)`。

注意 GA `ToolBase.Builder` 没有通用 `build()`；`ToolBase#callAsync` 还是一个默认返回运行时错误的 concrete 方法。项目新增 `AbstractPlatformAgentTool`，通过受控构造器调用 `super(builder)`，并把 `callAsync(ToolCallParam)` 重新声明为 abstract；具体工具必须 override，不能照抄上游错误 Javadoc 示例，也不能只靠“可编译”判断实现完整。

每个工具必须：

- schema 根为 object；
- `additionalProperties:false`；
- 显式 `readOnly`；
- 显式 `concurrencySafe`；
- 只从白名单注册；
- 从 RuntimeContext 取业务上下文；
- 保持现有工具名、参数 ID 和返回结构。

### 14.2 阻塞边界

- 真异步 HTTP/DB 优先使用响应式 API；
- 无法立即改造的同步 DB/HTTP/文件操作只能使用有容量、队列和拒绝策略的 `agent-tool-blocking` Scheduler，不直接依赖全局无界弹性队列；
- 生成轮询使用调度/`Mono.delay`，不使用 `Thread.sleep()`；
- Web/Reactor event-loop 中禁止 `.block()` 和 `.toIterable()`；
- 工具、生成任务和 Pipeline 使用一个统一 deadline，子层只能缩短，不能延长；
- 取消信号在调用前、执行中和业务回填前检查。
- 具有业务副作用的回填还必须校验 runId/ownerEpoch fencing token；仅检查内存布尔值不足以阻止失去 lease 的旧 owner。

### 14.3 平台子 Agent

- 保留 asset_image_gen、storyboard_frame_gen、storyboard_video_gen 等业务 Agent；
- 继续以平台工具形式暴露；
- parentToolCallId、agentName、childRunId 和父子取消关系稳定；
- 子 Agent 使用独立 RuntimeContext/StateStore session key；
- 项目、资产、子资产和分镜上下文从 RuntimeContext 或显式工具参数获取；
- 原生 subagent 的 taskId 不写入 childRunId。

## 15. 取消语义

### 15.1 取消请求

1. 校验 run 与当前用户归属；
2. 数据库 CAS `RUNNING/WAITING_CONFIRMATION/WAITING_EXTERNAL -> CANCEL_REQUESTED`；
3. 提交后写共享 cancel flag，并广播 runId；
4. owner 节点调用 `delegate.interrupt(runtimeContext)`；
5. dispose Agent/Provider Flux；
6. 调用可取消工具句柄；
7. 工具禁止在取消后回填资产、分镜或任务成功；
8. 通过统一 terminal coordinator 写 CANCELLED。

重复取消返回当前状态，是幂等操作。已 COMPLETED/FAILED 的 run 不伪装成 CANCELLED。

### 15.2 覆盖状态

取消测试覆盖：

- 等待执行；
- 模型连接建立中；
- 模型流式输出中；
- 工具执行中；
- 等待用户确认；
- 外部执行等待中；
- 子 Agent 执行中；
- 终态竞争；
- Redis 通知丢失；
- owner 节点失联。

## 16. 错误处理与可观测性

### 16.1 错误分类

- `MODEL_AUTH_FAILED`
- `MODEL_NOT_FOUND`
- `MODEL_PROTOCOL_ERROR`
- `MODEL_TIMEOUT`
- `TOOL_VALIDATION_FAILED`
- `TOOL_TIMEOUT`
- `TOOL_CANCELLED`
- `STATE_STORE_FAILED`
- `EVENT_PERSIST_FAILED`
- `AGENT_EVENT_BACKPRESSURE_OVERFLOW`
- `HARNESS_CAPACITY_EXHAUSTED`
- `RUN_CONFIG_UNAVAILABLE`
- `CONFIRMATION_EXPIRED`
- `EXTERNAL_EXECUTION_EXPIRED`
- `SSE_CURSOR_INVALID`
- `OWNER_LOST`
- `RUN_CANCELLED`
- `AGENTSCOPE_INTERNAL_ERROR`

ERROR SSE 只返回脱敏摘要和可追踪 code；详细异常进入服务端日志。

### 16.2 指标

至少记录：

- 活动 run、等待取消和失联 run 数；
- Harness cache size/hit/miss/eviction/active lease；
- StateStore 延迟和失败；
- AgentEvent 持久化延迟、sequence 冲突；
- Redis 未发布 outbox 数和重试次数；
- Provider complete/error/cancel 与未关闭资源；
- 工具 event-loop 违规和超时；
- 重连回放条数、重复过滤和终态恢复。

日志关联键统一使用 `runId/conversationId/userId/agentName/toolCallId`，敏感值脱敏。

## 17. 测试与验收

### 17.1 依赖与编译契约

- `dependency:tree` 所有 `io.agentscope` 为 `2.0.0`；
- 无 V1、RC、starter、session-mysql；
- 固定 HarnessAgent、RuntimeContext、AgentStateStore、streamEvents、ToolBase、ToolCallParam、ChatModelBase、User/AssistantMessage、URL/Base64Source；
- 编译验证 RuntimeContext null 处理、Harness delegate 定向 interrupt、ToolBase 无 build；
- 白名单扫描验证每个具体工具都 override `callAsync`，并对每个注册工具执行最小调用契约测试，避免继承 GA 默认错误实现。

### 17.2 Harness 与 StateStore

- 同一 Harness 不同 user/session 并发隔离；
- 64 个 cache slot 全部持有 lease 时，第 65 个 key 有界等待后 503，不突破容量或关闭 active entry；
- lease 获取后订阅前取消、complete/error/cancel 和双重 close 均恰好释放一次；
- 同一 conversation 活动 run 唯一；
- 重建 Harness 后 Redis 恢复；
- local/test InMemory 隔离；
- local 应用级 InMemory Store 跨 Harness 淘汰保留状态；测试 Context 之间不共享；
- conversation 删除后完整 session 清理；
- Redis 暂时不可用时错误明确，不静默退回内存；
- 模拟 GA 吞掉 load 异常，StateStore guard 仍必须在首个模型/工具调用前 fail-closed；
- 模拟 save 失败，run 不得进入 COMPLETED；
- adapter bulkhead/timeout 限制 GA 内部 boundedElastic 上的并发阻塞。

### 17.3 消息与媒体

- 文本、工具、工具结果不降级；
- URLSource/Base64Source、MIME 和顺序；
- `/media`、公网 URL、Data URI、Base64 转换；
- SSRF、私网、重定向、大小和 MIME；
- 不把 Base64 写入 DB/log。

### 17.4 AgentEvent 与 Journal

- 31 个 GA 枚举全部覆盖；
- 完整性测试以 `AgentEventType.values()` 和具体 subtype 映射表为准；不能用 `*Event.class` 文件数断言 31，因为还存在抽象 `AgentEvent` 基类；
- 新枚举导致测试失败；
- raw identity/createdAt/source 保留；
- 多线程上游事件仍按 journal 串行；
- `(runId, sequence)` 唯一且严格递增；
- committed chunk 后才发送；
- Redis 重复、乱序、裁剪和发布失败；
- DB replay/live 交界不重不漏；
- 在订阅、watermark、snapshot、tail query 每个窗口插入事件，均不遗漏；
- Redis 提示乱序、完全丢失和缓冲溢出时仍以 DB sequence 有序输出；
- MockWebTestClient 断言真实 SSE wire 同时包含 `id: runId:sequence` 和 `data:`，并验证 header/query 冲突返回 400；
- completed/error/cancel 后均可重放终态；
- outbox 补偿不会产生 UI 重复；
- 多实例 outbox claim、claim 超时接管、毒消息退避和积压告警；
- 消息投影崩溃重试不重复，并能推进到 terminal sequence；
- HTTP/SSE 断开后业务 run 继续，重连能看到断开期间事件和终态；
- AgentEvent ingress 事件数/字节超限时有界失败并释放 Provider，不发生无界内存增长。

### 17.5 终态与取消

- complete/error/cancel 三方竞态只产生一个终态；
- 终态事件恰好一个；用量结算至少一次重试，并通过 `(runId, modelCallId)` 或持久账本聚合后的 runId 幂等达到业务效果一次；
- 资源均关闭；
- 各模型/工具/确认/外部执行/子 Agent 阶段取消；
- 用户确认拒绝/批准、确认过期、重复确认和跨节点恢复；
- 外部执行结果 identity/权限校验和跨节点恢复；
- RequireUserConfirm candidate 到 StateStore save/AgentResult 之间崩溃时不暴露不可恢复确认按钮；
- TOOL_SUSPENDED 合成 WAITING_EXTERNAL，并用匹配 ToolResultBlock 恢复；
- WAITING 恢复严格使用持久 kernel fingerprint/snapshot；配置不可用或工具 manifest 不兼容时 fail closed；
- 取消后无 DONE、无业务成功回填；
- Redis 丢通知由 DB 状态补偿；
- owner epoch/fencing 阻止失去 lease 的旧节点继续副作用回填；
- owner lost 收敛为明确终态。

### 17.6 Provider

- OpenAI、Anthropic、Gemini、DashScope、Ollama 构造测试；
- MockWebServer 最小流式请求与解析；
- Responses/proxy/formatter/Vertex 定制契约；
- complete/error/cancel 关闭测试；
- 单个 ReAct run 多次 model call 时，调用级 usage 全部持久化、分别幂等结算并正确聚合；
- `doStream` 缺失 Reactor RuntimeContext 时在 Provider 调用前失败；
- 增加隔离的 `agentscope-provider-smoke` Maven profile，使用环境密钥执行五 Provider 真实最小流式 smoke；
- 增加隔离的 `ark-smoke` Maven profile，执行 Ark 图片/视频 smoke；
- 日志不泄漏密钥。

### 17.7 前端

- normalization schema 版本；
- 重复/乱序和 sequence gap；
- Replay/live 边界；
- 主/子 Agent ERROR/CANCELLED；
- 缺失 TOOL_CALL placeholder；
- 主终态 settle；
- 页面刷新后的 running/terminal 恢复；
- `corepack pnpm test`、`lint`、`build`。

### 17.8 多实例基础设施测试

增加 `agentscope-integration` Maven profile，使用 Failsafe + Testcontainers 启动 MySQL 8 和 Redis，并在同一测试中创建两个具有不同 instanceId 的 Coordinator/owner。它必须真实验证：

- Flyway Migration、generated active slot 和 message_order 唯一约束；
- RedisAgentStateStore 跨 Harness/实例恢复；
- owner epoch/fencing、lease 失效和 Redis 取消通知丢失；
- 多实例 outbox claim/接管；
- replay/watermark/tail-query；
- WAITING_CONFIRMATION 跨实例恢复。

这些性质不能只用 Mockito 证明。容器不可用时 profile 必须失败并报告环境问题，不能静默跳过。

### 17.9 最终命令

```powershell
cd D:\develop\my\ai-fusion-video\ai-fusion-video
.\mvnw.cmd dependency:tree -Dincludes=io.agentscope
.\mvnw.cmd test
.\mvnw.cmd package
.\mvnw.cmd -Pagentscope-integration verify
.\mvnw.cmd -Pagentscope-provider-smoke verify
.\mvnw.cmd -Park-smoke verify

cd D:\develop\my\ai-fusion-video\ai-fusion-video-web
corepack pnpm install --frozen-lockfile
corepack pnpm test
corepack pnpm lint
corepack pnpm build
```

任何 smoke 因缺失用户凭据不能运行时必须明确报告为未验证，不能写成通过；完整 P-1 验收仍要求在具备凭据的环境补跑。smoke profile 启动时先校验所需环境变量，缺失时快速失败，不静默跳过。

## 18. 分步实施与回滚

### P-1A-1：基线与依赖

- 修复既有测试夹具；
- 固定 Maven/pnpm 工具链；
- 加 GA 编译契约测试；
- 升级 AgentScope/Ark，收敛依赖树；
- 暂不切生产 Runtime。

### P-1A-2：最小 V2 Kernel

- StateStore 工厂；
- Harness 工厂/lease cache；
- RuntimeContext；
- 强类型文本消息；
- 跑通无工具 Agent 的 call/streamEvents。

### P-1B-1：run/event Journal 与兼容 SSE

- 新增增量 Migration；
- RunCoordinator/Journal/Mapper；
- DB-first 发布、outbox、reconnect；
- 前端 normalization/dedup；
- 保持旧 outputType。

### P-1B-2：模型

- 五官方扩展；
- 逐个迁移必要自定义模型；
- Provider 资源关闭和一次结算。

### P-1B-3：工具、媒体与子 Agent

- ToolBase；
- 阻塞边界和 deadline；
- URL/Base64 媒体；
- 平台子 Agent 父子关系。

### P-1B-4：取消、恢复与清理

- 跨节点取消；
- owner heartbeat/reconciliation；
- 三终态 CAS；
- 删除 V1 Hook/Session/Msg/Model；
- 全量回归。

### 回滚

- Migration 只新增表/列并稳定重排历史 message_order，不删除旧列、消息 ID 或业务数据；`next_message_order` 带默认值，旧二进制仍可插入 conversation；
- V2 Redis 使用独立 key prefix，不污染 V1；
- 回滚前先关闭 Pipeline 写入口，取消或终态化活动 V2 run，停止 owner，处理或保留待发 outbox，再启动 V1；不能在活跃 V2 run 存在时直接替换二进制；
- 回滚应用二进制但不回滚已执行 Migration；V1 回滚期间可能继续按旧 MAX+1 写消息，因此再次启用 V2 前必须在停写窗口执行 `next_message_order = GREATEST(next_message_order, MAX(全部消息含 deleted)+1)` 对账；
- 首次 Migration 和 V1/V2 切换均使用短维护写入窗口；不承诺旧 V1 与新 V2 消息 writer 的无协调滚动混跑；
- 不在 V1/V2 间双写同一 AgentState；
- V2 上线验证完成后立即删除生产 V1 Runtime 选择，避免长期分叉。

## 19. 预计文件边界

### 后端现有文件

- `ai-fusion-video/pom.xml`
- `AgentScopeAssistantService.java`
- `StreamingEventHook.java`（最终删除）
- `AgentScopeToolAdapter.java`
- `AgentScopeSubAgentToolAdapter.java`
- `AgentScopeModelFactory.java`
- 各 Provider AgentScope 适配器
- `AiStreamRedisService.java`
- `AgentMessageService.java`
- `AiChatStreamRespVO.java`
- `AiPipelineController.java`

### 后端新增边界

- `AgentScopeHarnessFactory`
- `HarnessLeaseCache`
- `AgentScopeRuntimeContextFactory`
- `AgentScopeStateStoreFactory`
- `FailClosedAgentStateStore` / `StateStoreGuardedChatModel`
- `AgentScopeMessageMapper`
- `AgentScopeEventMapper`
- `AgentEventEnvelopeSanitizer`
- `AgentRunCoordinator`
- `RunExecutionSupervisor`
- `RunTerminalCoordinator`
- `AgentEventJournal`
- `AgentRuntimeSchedulers`
- `ModelCallUsageLedger`
- `CancellationCoordinator`
- `AbstractPlatformAgentTool` / `RunLeaseGuard`
- run/event/model-call-usage entity、mapper、repository
- `V1.0.6.1.5__agent_run_and_event.sql`

### 前端

- `lib/api/ai-pipeline.ts`
- `lib/api/ai-assistant.ts`
- `lib/store/pipeline-store.ts`
- 新增 event normalization/dedup 模块及测试
- 保持通知面板、history 和结果 renderer 的现有结构

## 20. 官方核验来源

核验日期：2026-07-21；目标版本：AgentScope Java `2.0.0` GA。

- https://github.com/agentscope-ai/agentscope-java/releases/tag/v2.0.0
- https://github.com/agentscope-ai/agentscope-java/commit/44c304ec84d5fbd8588c1af8bc71b1edb9663380
- https://repo1.maven.org/maven2/io/agentscope/agentscope-core/2.0.0/agentscope-core-2.0.0-sources.jar
- https://repo1.maven.org/maven2/io/agentscope/agentscope-harness/2.0.0/agentscope-harness-2.0.0-sources.jar
- https://repo1.maven.org/maven2/io/agentscope/agentscope-extensions-redis/2.0.0/agentscope-extensions-redis-2.0.0-sources.jar
- `D:\develop\my\ai-work-studio\docs\agentscope-v2\README.md`
- `D:\develop\my\ai-work-studio\docs\agentscope-v2\2026-07-10-rc4-to-ga-change-audit.md`

模型能力资料仍须在实际 Provider 迁移时按具体模型版本查厂商官方文档并记录 URL、核验日和版本；本规格不依据模型名称推测媒体能力。

## 21. 完成定义

P-1 只有在以下条件全部有真实证据时完成：

1. 依赖树无 V1/RC/混合 AgentScope；
2. 生产入口全部使用 Harness + RuntimeContext + AgentStateStore；
3. 强类型消息、媒体、31 类事件、工具和子 Agent 契约通过；
4. 数据库 run/event/message 顺序和回放验证通过；
5. complete/error/cancel 资源释放且终态事件恰好一个；费用结算至少一次重试，并以 `(runId, modelCallId)` 或持久账本聚合后的 runId 幂等达到业务效果一次；
6. 跨节点取消和丢通知补偿通过；
7. 五 Provider、必要自定义能力和 Ark smoke 通过；
8. 后端全量测试与 package 通过；
9. 前端 test/lint/build 通过；
10. V1 Hook、Session、Msg、Model 和无效依赖全部移除；
11. 没有用“理论可用”替代未执行验证。

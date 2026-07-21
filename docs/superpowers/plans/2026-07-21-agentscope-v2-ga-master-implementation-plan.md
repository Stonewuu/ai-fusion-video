# AgentScope Java 2.0.0 GA 迁移总控 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在当前 `main` 上把 ai-fusion-video 从 AgentScope Java 1.0.12 完整迁移到 2.0.0 GA，并以可恢复、可审计、多实例一致的 Durable Run 取代 conversationId 驱动的 V1 Runtime。

**Architecture:** 迁移按依赖/Kernel、Durable Runtime、模型/工具/媒体、前端/切换四个纵向计划执行，每个阶段都先建立失败测试、再实现最小可交付切片、最后通过独立 review gate。MySQL run/event journal 是执行与回放真相，Redis 分别承担 AgentStateStore 和 committed-event wake-up；服务端 supervisor 独立拥有业务订阅，浏览器只是可重连观察者。

**Tech Stack:** Java 21、Spring Boot 3.5、Project Reactor、AgentScope Java 2.0.0 GA、MyBatis-Plus、Flyway、MySQL 8、Redis、Testcontainers/Failsafe、Next.js 16、React 19、TypeScript、Zustand、Vitest、pnpm 10.32.1。

## Global Constraints

- 设计真相为 `docs/superpowers/specs/2026-07-21-agentscope-v2-ga-migration-design.md`，状态必须保持“书面规格已确认”。
- 只在当前 `main` 上纵向迁移，不合并 `origin/feat-agentscope-v2`，不长期保留 V1/V2 双运行时。
- 所有 AgentScope 模块统一锁定 `2.0.0` GA；删除 starter、session-mysql、legacy runtime bridge 和已被 GA extension 替代的兼容实现。GA 保留的重叠 FQN 不能仅凭名称判作 V1。
- 五个官方 model extension 必须全部保留：OpenAI、Anthropic、Gemini、DashScope、Ollama。
- 生产状态存储使用 Spring 管理的共享 `RedisAgentStateStore`；local/test 使用应用级共享、测试上下文隔离的 `InMemoryAgentStateStore`。
- Redis AgentStateStore key prefix 使用 `afv:agentscope:v2:`；V2 不污染 V1 状态。
- Harness cache 硬容量为 64；64 个不同 active key 占满时第 65 个最多等待 5 秒后返回 `HARNESS_CAPACITY_EXHAUSTED`/503，不能突破容量或关闭 active entry。
- AgentEvent ingress 默认最多 4096 个事件或 8MiB；任一超限都以 `AGENT_EVENT_BACKPRESSURE_OVERFLOW` 有界失败。
- owner heartbeat 每 5 秒、lease 20 秒、reconcile 每 5 秒；所有租约判断使用数据库时间和 owner epoch/fencing。
- MySQL 是回放真相；Redis live/outbox payload 只作 wake-up，不能直接作为顺序真相。
- 每 run 的 `sequence_no` 严格递增并受 `UNIQUE(run_id,sequence_no)` 保护；complete/error/cancel 终态事件恰好一个。
- 模型用量按调用生成 `modelCallId`，受 `UNIQUE(run_id,model_call_id)` 保护；至少一次重试加下游幂等后实现业务效果一次。
- 标准 SSE `id` 是 `{runId}:{sequence}`；断线不取消业务 run，EOF 不能推断完成。
- 确认按钮只能在 StateStore 保存与暂停 AgentResult 成功后发布；外部等待由 `TOOL_SUSPENDED` 合成并以匹配 `ToolResultBlock` 恢复。
- 平台子 Agent 的 `childRunId` 与 AgentScope 原生 `taskId` 不得混用；P-1 默认不启用原生 subagent。
- 每个 conversation 最多一个活动 root run；child 共享父 conversation，但 `UNIQUE(parent_run_id,parent_tool_call_id)` 保证工具重试幂等，且 child 不占 root 活动槽。
- run 的绝对 `deadline_at` 必须持久化；WAITING 恢复、Provider、工具和 child 只能继承或缩短，不能按恢复时刻重新延长。
- 不修改已执行 Flyway，只新增 `V1.0.6.1.5__agent_run_and_event.sql`；保留资产、子资产、分镜、任务、conversation/message ID。
- AgentScope/Reactor 链路禁止 `.block()`、`.toIterable()`、`Thread.sleep()`、ThreadLocal 和未隔离的阻塞 I/O。
- 模型能力只依据厂商官方资料，记录 URL、核验日 `2026-07-21` 和具体模型版本。
- 最终必须实跑后端全量、MySQL/Redis 多实例集成、Provider/Ark smoke、前端 test/lint/build；缺少凭据只能记为未验证。

---

## Sub-plan map and frozen boundaries

1. `docs/superpowers/plans/2026-07-21-agentscope-v2-ga-dependency-kernel-implementation-plan.md`
   - 产出：绿色基线、2.0.0 GA 依赖、四个有界 scheduler、RuntimeContext、共享 fail-closed StateStore、Harness factory 与 hard-cap lease cache。
2. `docs/superpowers/plans/2026-07-21-agentscope-v2-ga-durable-runtime-implementation-plan.md`
   - 产出：Migration、run/event/usage repository、owned/system journal/terminal、outbox/projection、owner fencing、cancel、bounded supervisor、持久化平台 child run、snapshot/deadline 恢复、replay-live、标准 SSE 后端。
3. `docs/superpowers/plans/2026-07-21-agentscope-v2-ga-model-tool-media-implementation-plan.md`
   - 产出：调用级 usage、五 Provider、自定义 Responses、媒体、可取消 ToolBase、Harness tool registry、平台子 Agent adapter、WAITING HTTP 恢复、Provider/Ark smoke。
4. `docs/superpowers/plans/2026-07-21-agentscope-v2-ga-frontend-cutover-implementation-plan.md`
   - 产出：前端 parser/normalizer/cursor/store、逐工具确认 UI、legacy runtime bridge 删除、切换/回滚手册、最终验证证据。

冻结的跨计划接口：

```java
public interface RunLeaseGuard {
    Mono<Void> assertLease(String runId, String ownerInstanceId, long ownerEpoch);
}

public interface AgentRuntimeShutdownPort {
    Mono<Void> shutdown(Duration drainTimeout);
}

public record StartAgentExecutionCommand(
    StartedAgentRun run,
    List<Msg> messages,
    AgentKernelSnapshot kernelSnapshot,
    AgentKernelSpec kernelSpec,
    AgentScopeRuntimeContextRequest runtimeContextRequest) {}

public record ResumeAgentExecutionCommand(
    ResumedAgentRun run,
    List<Msg> messages,
    AgentKernelSnapshot kernelSnapshot,
    AgentScopeRuntimeContextRequest runtimeContextRequest) {}

public interface RunExecutionSupervisor extends AgentRuntimeShutdownPort {
    Mono<Void> start(StartAgentExecutionCommand command);
    Mono<Void> resume(ResumeAgentExecutionCommand command);
    Mono<Boolean> interruptOwned(String runId, String ownerInstanceId,
        long ownerEpoch, ExecutionStopReason reason);
}

public interface CancellationCoordinator {
    Mono<AgentRunStatus> cancel(String runId, long currentUserId);
}

public interface AgentEventJournal {
    Mono<Optional<CommittedAgentEvent>> appendOwned(String runId,
        String ownerInstanceId, long ownerEpoch, AgentEventEnvelope event);
}

public interface RunTerminalCoordinator {
    Mono<Optional<CommittedAgentEvent>> terminateOwned(RunTerminalRequest request,
        String ownerInstanceId, long ownerEpoch);
    Mono<Optional<CommittedAgentEvent>> terminateSystem(RunTerminalRequest request,
        SystemTerminalActor actor);
}

public interface ModelCallUsageLedger {
    Mono<ModelCallTicket> start(AgentRunContext run, String provider, String modelCode);
    Mono<Void> complete(ModelCallTicket ticket, NormalizedModelUsage usage);
    Mono<Void> fail(ModelCallTicket ticket, Throwable failure);
    Mono<Void> cancel(ModelCallTicket ticket);
}

public interface ModelUsageSettlementPort {
    Mono<String> settle(String idempotencyKey, NormalizedModelUsage usage);
}

public interface PlatformSubAgentRunPort {
    Mono<PlatformSubAgentRun> start(PlatformSubAgentCommand command);
    Mono<Void> cancelChildren(String parentRunId);
}

public interface AgentKernelSnapshotBuilder {
    AgentKernelSnapshot build(AgentKernelSpec spec);
}

public interface AgentKernelSnapshotResolver {
    Mono<AgentKernelSpec> resolve(AgentKernelSnapshot snapshot);
}

public interface AgentWaitingStatePort {
    Mono<Void> recordConfirmationCandidate(String runId, PendingConfirmation candidate);
    Mono<Boolean> enterWaitingConfirmation(String runId, long expectedOwnerEpoch, WaitingCheckpoint checkpoint);
    Mono<Boolean> enterWaitingExternal(String runId, long expectedOwnerEpoch, WaitingCheckpoint checkpoint, PendingExternalExecution pending);
    Mono<PendingConfirmation> getPendingConfirmationAuthorized(String runId, long currentUserId, String replyId);
    Mono<PendingExternalExecution> getPendingExternalAuthorized(String runId, long currentUserId, String toolCallId);
    Mono<ResumedAgentRun> resumeConfirmation(ResumeConfirmationCommand command);
    Mono<ResumedAgentRun> resumeExternal(ResumeExternalCommand command);
}
```

### Task 1: 建立隔离执行环境和确认基线

**Files:**
- Read: `docs/superpowers/specs/2026-07-21-agentscope-v2-ga-migration-design.md`
- Read: `docs/superpowers/plans/2026-07-21-agentscope-v2-ga-dependency-kernel-implementation-plan.md`
- Modify only through sub-plans: production and test files.

**Interfaces:**
- Consumes: 当前 `main`、已确认规格、干净工作树。
- Produces: 一个通过 `superpowers:using-git-worktrees` 创建或确认的隔离工作区，以及可追溯的基线测试结果。

- [ ] **Step 1: 使用 worktree skill 检测当前 git 形态**

Run: `git rev-parse --git-dir; git rev-parse --git-common-dir; git branch --show-current; git status --short --branch`

Expected: 能判定是否已在 linked worktree；工作树除本计划授权文件外无未知改动。

- [ ] **Step 2: 若不在隔离 worktree，按 skill 创建 `codex/agentscope-v2-ga`**

Run: `git worktree add ..\ai-fusion-video-agentscope-v2 -b codex/agentscope-v2-ga main`

Expected: 新工作区创建成功，分支名为 `codex/agentscope-v2-ga`；如果当前环境已经是 app 管理的 worktree，则记录“无需创建”而不嵌套 worktree。

- [ ] **Step 3: 记录 Java、Maven Wrapper 和 pnpm 版本**

Run: `java -version; javac -version; cd ai-fusion-video; .\mvnw.cmd -version; cd ..\ai-fusion-video-web; corepack pnpm --version`

Expected: 当前环境记录 Temurin Java/Javac 21.0.11（或兼容 Java 21 patch），Maven Wrapper 3.9.12，pnpm 10.32.1。

- [ ] **Step 4: 复现并保存迁移前后端基线**

Run: `cd D:\develop\my\ai-fusion-video\ai-fusion-video; .\mvnw.cmd test`

Expected: 在修复 fixture 前复现 `GenerationModelCapabilityServiceTests` 与 `GetGenerationModelCapabilitiesToolExecutorTests` 的 7 failures/10 errors；不得把它们归因于 AgentScope。

Run: `cd D:\develop\my\ai-fusion-video\ai-fusion-video-web; corepack pnpm lint; corepack pnpm build`

Expected: 记录实际退出码和既有告警；不在本步骤顺手修复无关问题。

### Task 2: 执行依赖与 V2 Kernel 计划

**Files:**
- Execute exactly: `docs/superpowers/plans/2026-07-21-agentscope-v2-ga-dependency-kernel-implementation-plan.md`

**Interfaces:**
- Consumes: Task 1 基线。
- Produces: `AgentRuntimeSchedulers`、`AgentScopeRuntimeContextFactory`、`FailClosedAgentStateStore`、`AgentScopeHarnessFactory`、`HarnessLeaseCache` 与最小 V2 event stream。

- [ ] **Step 1: 使用 `superpowers:subagent-driven-development` 为该子计划逐 task 派发新 implementer**

Expected: 每个 task 有独立 implementer、规格 review、代码质量 review；reviewer 不与 implementer 复用上下文。

- [ ] **Step 2: 每个 task 完成后执行其定向测试命令并提交**

Expected: 每次提交只覆盖一个可独立审查交付物；失败测试证据先于生产实现。

- [ ] **Step 3: 运行 Phase Gate A**

Run: `cd D:\develop\my\ai-fusion-video\ai-fusion-video; .\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true "-Dtest=AgentScopeGaDependencyContractTests,AgentScopeGaApiContractTests,AgentRuntimeSchedulersTests,AgentScopeRuntimeContextFactoryTests,AgentScopeStateStoreFactoryTests,StateStoreFailureGuardTests,FailClosedAgentStateStoreTests,AgentStatePreflightTests,HarnessLeaseCacheTests,AgentScopeHarnessFactoryTests,DefaultAgentScopeHarnessInvokerTests,AgentScopeMessageMapperTests,AgentRuntimeShutdownPortTests,AgentScopeKernelLifecycleTests" test`

Expected: PASS；第 65 个 cache key 的 503、save failure 禁止 COMPLETED、同 Harness 跨 user/session 隔离均有测试。

- [ ] **Step 4: 冻结 Kernel API，后续计划不得改名绕开 review**

Run: `git grep -n "class HarnessLeaseCache\|interface AgentScopeHarnessFactory\|class FailClosedAgentStateStore\|class AgentRuntimeSchedulers"`

Expected: 每个冻结类型只有一个生产定义；不存在备用 V1/V2 factory。

### Task 3: 执行 Durable Runtime 计划

**Files:**
- Execute exactly: `docs/superpowers/plans/2026-07-21-agentscope-v2-ga-durable-runtime-implementation-plan.md`

**Interfaces:**
- Consumes: Task 2 Kernel、四个有界 scheduler、共享 StateStore。
- Produces: durable run/event/usage 数据模型、DB-first journal、terminal CAS、projection、outbox、fencing/cancel、supervisor、replay-live 和后端 SSE。

- [ ] **Step 1: 在任何生产 repository 前先执行 Migration 失败测试**

Run: `cd D:\develop\my\ai-fusion-video\ai-fusion-video; .\mvnw.cmd -Pagentscope-integration "-Dit.test=AgentPersistenceMigrationIT" verify`

Expected: Migration 实现前 FAIL，原因是新表/生成列/唯一约束不存在，而不是容器被跳过。

- [ ] **Step 2: 按 durable 子计划逐 task 执行 implementer + 两阶段 review**

Expected: run/event/outbox/projection/cancel/replay 各自拥有独立提交和定向测试。

- [ ] **Step 3: 运行 Phase Gate B 单元测试**

Run: `.\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true "-Dtest=BoundedAgentEventIngressTests,AgentEventChunkCoalescerTests,RunExecutionSupervisorTests,AgentRuntimeMetricsTests,AiPipelineSseControllerTests,DurableRuntimeRequiredTestsContractTests" test`

Expected: PASS；终态恰好一个、EOF 不取消业务执行、cursor 冲突 400、Redis 丢取消通知由 DB 补偿。

- [ ] **Step 4: 运行 Phase Gate B 多实例测试**

Run: `.\mvnw.cmd -Pagentscope-integration -Dfailsafe.failIfNoSpecifiedTests=true "-Dit.test=AgentPersistenceMigrationIT,AgentRunStartIT,AgentMessageAllocatorIT,AgentJournalTerminalIT,AgentModelCallUsageIT,AgentOutboxMultiInstanceIT,AgentProjectionRecoveryIT,PlatformSubAgentRunServiceIT,AgentWaitingStateIT,AgentFencingCancellationIT,AgentOwnedJournalTakeoverIT,AgentReplayLiveIT,AgentDurableRuntimeMultiInstanceIT" verify`

Expected: PASS 且 Failsafe 汇总 `Skipped: 0`；MySQL 8/Redis 容器真实启动，两个不同 instanceId 验证 fencing、outbox claim、replay-live、child start/cancel 竞态、绝对 deadline 和 WAITING 跨节点恢复。

### Task 4: 执行模型、工具、媒体与 WAITING 计划

**Files:**
- Execute exactly: `docs/superpowers/plans/2026-07-21-agentscope-v2-ga-model-tool-media-implementation-plan.md`

**Interfaces:**
- Consumes: `AgentModelCallUsageRepository`、`RunLeaseGuard`、`AgentWaitingStatePort`、Kernel factory。
- Produces: 五个 GA Provider、自定义 Responses `doStream`、调用级 usage、媒体映射、V2 tools/sub-agent、WAITING resume 和 smoke profiles。

- [ ] **Step 1: 锁定所有模型调用从 Reactor Context 取 RuntimeContext**

Run: `git grep -n "AgentBase.RUNTIME_CONTEXT_KEY" ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai`

Expected: 自定义 `ChatModelBase#doStream` 与统一 wrapper 使用 `Flux.deferContextual`；不存在 ThreadLocal fallback。

- [ ] **Step 2: 按模型/工具子计划逐 task 执行 implementer + 两阶段 review**

Expected: Provider 逐个迁移，GA 已覆盖的 proxy/formatter 补丁只在等价测试通过后删除。

- [ ] **Step 3: 运行 Phase Gate C**

Run: `cd D:\develop\my\ai-fusion-video\ai-fusion-video; .\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true "-Dtest=DefaultModelCallUsageLedgerTests,ModelCallUsageSettlementWorkerTests,ManagedAgentScopeChatModelTests,AgentScopeGaProviderFactoryTests,OpenAiResponsesAgentScopeModelTests,ProviderCapabilityEvidenceTests,SafeAgentMediaResolverTests,AgentScopeMessageMapperTests,PlatformToolRegistryTests,AgentScopeToolHarnessIntegrationTests,AgentScopeToolAdapterTests,ImageGenerationConsumerTests,VideoGenerationConsumerTests,AgentScopeSubAgentToolAdapterTests,AgentScopeSubAgentWiringTests,WaitingCoordinatorTests,AiPipelineWaitingControllerTests" test`

Expected: PASS；同一 run 两次 ReAct model call 写两个 modelCallId，complete/error/cancel 都关闭资源，工具不占 event-loop，WAITING 保存时序正确。

Run: `.\mvnw.cmd -Pagentscope-integration -Dfailsafe.failIfNoSpecifiedTests=true "-Dit.test=AgentToolCancellationIT,PlatformSubAgentRunServiceIT,AgentWaitingHttpResumeIT" verify`

Expected: PASS 且 `Skipped: 0`；取消/超时/fence 丢失抑制迟到成功，平台 child 身份幂等，HTTP 确认在另一节点按原 snapshot/deadline 恢复。

- [ ] **Step 4: 只在具备凭据时运行 smoke，并保留失败证据**

Run: `.\mvnw.cmd -Pagentscope-provider-smoke verify`

Expected: 五 Provider PASS；缺凭据时 profile 快速失败，状态记为未验证。

Run: `.\mvnw.cmd -Park-smoke verify`

Expected: Ark 图片/视频 PASS；缺少 Ark 凭据时 profile 必须快速失败，并在验证文档中明确记为“未验证”，不得写成 PASS。

### Task 5: 执行前端兼容、V1 切换与操作手册计划

**Files:**
- Execute exactly: `docs/superpowers/plans/2026-07-21-agentscope-v2-ga-frontend-cutover-implementation-plan.md`

**Interfaces:**
- Consumes: 后端标准 SSE、run status/cancel/reconnect API，以及已完成的模型/工具 runtime。
- Produces: cursor-aware 前端、V1 代码删除、切换/回滚 runbook、最终验证文档。

- [ ] **Step 1: 按前端子计划执行 parser → normalizer → API → store → hook 顺序**

Expected: 每层先有独立失败测试；store 不直接解析 wire，parser 不持有业务状态。

- [ ] **Step 2: 运行前端 Phase Gate**

Run: `cd D:\develop\my\ai-fusion-video\ai-fusion-video-web; corepack pnpm test`

Expected: PASS；覆盖 CR/LF 恰跨 chunk、unknown output/raw/control 先于 cursor 失败、duplicate/乱序/gap、子终态、EOF、placeholder、invalidation、逐工具确认、过期和重复响应。

Run: `corepack pnpm lint; corepack pnpm build`

Expected: 两条均 PASS。

- [ ] **Step 3: 删除 V1 后运行静态扫描**

Run: `cd D:\develop\my\ai-fusion-video; rg -n "agentscope-spring-boot-starter|agentscope-extensions-session-mysql|io\.agentscope\.core\.session\.|MysqlSession|StreamingEventHook|AnthropicAgentScopeProxySupport|ProxyAwareAnthropicChatModel|GeminiToolResponseAwareChatFormatter|VertexAgentScopeProxySupport" ai-fusion-video/pom.xml ai-fusion-video/src/main ai-fusion-video/src/test`

Expected: 无生产旧 artifact/session/bridge 命中；GA 保留的重叠 FQN `Msg/Model/ReActAgent/AgentTool/ToolCallParam/core.hook.*` 不作为 V1 证据。

- [ ] **Step 4: 对照操作手册做一次非生产 dry-run**

Expected: preflight、停写、active run 收敛、outbox 检查、二进制回滚和 V2 重新启用对账均可逐条执行，不依赖口头知识。

### Task 6: 全规格自审、最终验证与完成判定

**Files:**
- Read: `docs/superpowers/specs/2026-07-21-agentscope-v2-ga-migration-design.md`
- Modify: `docs/verification/2026-07-21-agentscope-v2-ga-verification.md`

**Interfaces:**
- Consumes: Tasks 1-5 的同一最终 commit。
- Produces: P-1 完成或明确未完成的证据结论。

- [ ] **Step 1: 使用 `superpowers:requesting-code-review` 做规格符合性 review**

Expected: reviewer 逐项核对规格第 21 节完成定义、Migration/回滚安全、反应式禁令、资源关闭和测试证据；所有 Critical/High 问题修复后重新 review。

- [ ] **Step 2: 使用 `superpowers:verification-before-completion` 执行完整命令矩阵**

```powershell
cd D:\develop\my\ai-fusion-video\ai-fusion-video
.\mvnw.cmd dependency:tree "-Dincludes=io.agentscope"
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

Expected: 除明确缺凭据而记为未验证的 smoke 外全部 PASS；P-1 不能在强制 smoke 未补跑前宣称完全完成。

- [ ] **Step 3: 检查工作树和提交边界**

Run: `git status --short; git log --oneline --decorate -25`

Expected: 工作树干净；提交按测试基线、Kernel、durable runtime、Provider、tools、frontend、cutover、verification 分层，无无关用户文件。

- [ ] **Step 4: 使用 `superpowers:finishing-a-development-branch` 选择集成方式**

Expected: 只在测试与 review 全部完成后提供 merge/PR/保留分支选项；不自动删除用户工作。
